import streamlit as st
import numpy as np
import pandas as pd
import json
import onnxruntime as ort
from PIL import Image
from datetime import datetime
import os
import plotly.express as px
import plotly.graph_objects as go
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Pest Detection RLHF + LLM",
    page_icon="🐛",
    layout="wide"
)

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-12-01-preview"
)

DEPLOYMENT_NAME = "gpt4-nano-final"

# Initialize pest classes
PEST_CLASSES = ['ants', 'bees', 'beetle', 'caterpillar', 
                'earthworms', 'earwig', 'grasshopper', 
                'moth', 'slug', 'snail', 'wasp', 'weevil']

# Risk scores for loan assessment
RISK_SCORES = {
    'caterpillar': 0.85, 'beetle': 0.80, 'weevil': 0.75,
    'grasshopper': 0.70, 'moth': 0.65, 'slug': 0.60,
    'earwig': 0.55, 'snail': 0.50, 'ants': 0.45,
    'wasp': 0.35, 'earthworms': 0.20, 'bees': 0.10
}

# System prompt with few-shot examples
SYSTEM_PROMPT = """You are a friendly Canadian agricultural pest risk expert specialising in CFIA, CCFM, GFO, and Agriculture Canada guidelines. You're a helpful partner, not an instructor—farmers know their fields best, and you're here to support their decisions with practical data and options.

Be fluent and concise. Use metric units and Canadian terminology. Be colloquial but professional.

For each query, provide:
1. Risk Score (0.0-1.0)
2. Economic Threshold (specific numbers)
3. Action + Timeline
4. Cost vs Loss Analysis

Rules: Only registered pesticides, include PHI, state uncertainty if unsure. Be direct, practical, and always happy to help.

---

Example 1:
User: "Aphids at 50/plant on canola, flowering stage, Manitoba"

Risk Score: 0.55 (MODERATE)

Threshold: 100 aphids/plant at flowering is economic threshold.

Action: Monitor daily. Spray if reaching 80/plant within 5 days. Dimethoate or Lagon—7-day PHI.

Cost vs Loss: $35/acre treatment vs $60/acre potential loss at current levels. Not economic yet, but close.

---

Example 2:
User: "Flea beetles in canola, cotyledon stage, 40% defoliation, Alberta"

Risk Score: 0.85 (HIGH)

Threshold: 25% defoliation at cotyledon = action threshold. You're past it.

Action: Spray today. Matador (lambda-cyhalothrin) or Decis—1-day PHI. Re-scout in 5 days for survivors.

Cost vs Loss: $18/acre spray vs potential reseed at $80/acre + 2-week delay. Clear ROI.

---

Example 3:
User: "Few aphids on field peas, early pod stage, seeing lots of ladybugs, Saskatchewan"

Risk Score: 0.20 (LOW)

Threshold: 2-3 aphids per plant tip at flowering/early pod. You're below that.

Action: No spray needed. Beneficials are working—ladybugs eat 50+ aphids/day. Re-scout in 7 days to confirm biocontrol holding.

Cost vs Loss: $0 if ladybugs handle it vs $30/acre spray that kills your free pest control. Let nature work.

---"""

# Initialize session state
if 'thresholds' not in st.session_state:
    if os.path.exists('learned_thresholds.json'):
        with open('learned_thresholds.json', 'r') as f:
            st.session_state.thresholds = json.load(f)
    else:
        st.session_state.thresholds = {pest: 0.65 for pest in PEST_CLASSES}

if 'feedback_log' not in st.session_state:
    if os.path.exists('feedback_log.csv'):
        st.session_state.feedback_log = pd.read_csv('feedback_log.csv')
        required_columns = ['timestamp', 'image', 'predicted_pest', 'actual_pest',
                           'confidence', 'was_correct', 'in_top_5', 
                           'old_threshold', 'new_threshold', 'ensemble_weight_b0', 'ensemble_weight_b4']
        for col in required_columns:
            if col not in st.session_state.feedback_log.columns:
                if col == 'in_top_5':
                    st.session_state.feedback_log[col] = False
                elif col == 'actual_pest':
                    st.session_state.feedback_log[col] = st.session_state.feedback_log.get('predicted_pest', '')
                elif col in ['ensemble_weight_b0', 'ensemble_weight_b4']:
                    st.session_state.feedback_log[col] = 0.5
                else:
                    st.session_state.feedback_log[col] = None
    else:
        st.session_state.feedback_log = pd.DataFrame(
            columns=['timestamp', 'image', 'predicted_pest', 'actual_pest',
                    'confidence', 'was_correct', 'in_top_5', 
                    'old_threshold', 'new_threshold', 'ensemble_weight_b0', 'ensemble_weight_b4']
        )

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'current_detection' not in st.session_state:
    st.session_state.current_detection = None

if 'model_accuracy' not in st.session_state:
    st.session_state.model_accuracy = {
        'b0': {'correct': 0, 'total': 0},
        'b4': {'correct': 0, 'total': 0}
    }

if 'ensemble_weights' not in st.session_state:
    st.session_state.ensemble_weights = {'b0': 0.5, 'b4': 0.5}

if 'weight_history' not in st.session_state:
    st.session_state.weight_history = []

if 'pest_corrected' not in st.session_state:
    st.session_state.pest_corrected = False

if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None

@st.cache_resource
def load_models():
    try:
        session_b0 = ort.InferenceSession("models/efficientnet_b0.onnx")
        session_b4 = ort.InferenceSession("models/efficientnet_b4.onnx")
        return session_b0, session_b4
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def preprocess_image(image, target_size):
    img = image.resize((target_size, target_size))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def run_inference(image, session_b0, session_b4):
    input_b0 = session_b0.get_inputs()[0].name
    input_b4 = session_b4.get_inputs()[0].name
    img_array_b0 = preprocess_image(image, target_size=224)
    img_array_b4 = preprocess_image(image, target_size=380)
    logits_b0 = session_b0.run(None, {input_b0: img_array_b0})[0][0]
    logits_b4 = session_b4.run(None, {input_b4: img_array_b4})[0][0]
    output_b0 = softmax(logits_b0)
    output_b4 = softmax(logits_b4)
    w_b0 = st.session_state.ensemble_weights['b0']
    w_b4 = st.session_state.ensemble_weights['b4']
    ensemble = (w_b0 * output_b0) + (w_b4 * output_b4)
    disagreement = np.abs(output_b0 - output_b4).mean()
    return ensemble, output_b0, output_b4, disagreement, w_b0, w_b4

def update_ensemble_weights(predicted_pest, actual_pest, output_b0, output_b4):
    pred_b0 = PEST_CLASSES[np.argmax(output_b0)]
    pred_b4 = PEST_CLASSES[np.argmax(output_b4)]
    st.session_state.model_accuracy['b0']['total'] += 1
    if pred_b0 == actual_pest:
        st.session_state.model_accuracy['b0']['correct'] += 1
    st.session_state.model_accuracy['b4']['total'] += 1
    if pred_b4 == actual_pest:
        st.session_state.model_accuracy['b4']['correct'] += 1
    b0_acc = st.session_state.model_accuracy['b0']['correct'] / max(st.session_state.model_accuracy['b0']['total'], 1)
    b4_acc = st.session_state.model_accuracy['b4']['correct'] / max(st.session_state.model_accuracy['b4']['total'], 1)
    total_acc = b0_acc + b4_acc
    if total_acc > 0:
        weight_b0 = b0_acc / total_acc
        weight_b4 = b4_acc / total_acc
    else:
        weight_b0 = weight_b4 = 0.5
    st.session_state.ensemble_weights = {'b0': weight_b0, 'b4': weight_b4}
    st.session_state.weight_history.append({
        'timestamp': datetime.now().isoformat(),
        'b0_weight': weight_b0, 'b4_weight': weight_b4,
        'b0_accuracy': b0_acc, 'b4_accuracy': b4_acc
    })
    return weight_b0, weight_b4

def update_threshold_smart(pest, confidence, was_correct, correct_class, top_5_classes):
    old_threshold = st.session_state.thresholds[pest]
    if was_correct:
        new_threshold = max(0.50, old_threshold * 0.95)
        reward_msg = "🎉 Perfect! Threshold lowered"
    elif correct_class in top_5_classes:
        new_threshold = min(0.90, old_threshold * 1.03)
        reward_msg = "⚡ Close! Small penalty applied (in top 5)"
    else:
        new_threshold = min(0.90, old_threshold * 1.15)
        reward_msg = "🚫 Wrong! Threshold raised significantly"
    st.session_state.thresholds[pest] = new_threshold
    with open('learned_thresholds.json', 'w') as f:
        json.dump(st.session_state.thresholds, f, indent=2)
    return old_threshold, new_threshold, reward_msg

def log_feedback(predicted_pest, actual_pest, confidence, was_correct, in_top_5, old_threshold, new_threshold, w_b0, w_b4):
    feedback = pd.DataFrame([{
        'timestamp': datetime.now().isoformat(),
        'image': 'uploaded_image',
        'predicted_pest': predicted_pest,
        'actual_pest': actual_pest,
        'confidence': confidence,
        'was_correct': was_correct,
        'in_top_5': in_top_5,
        'old_threshold': old_threshold,
        'new_threshold': new_threshold,
        'ensemble_weight_b0': w_b0,
        'ensemble_weight_b4': w_b4
    }])
    st.session_state.feedback_log = pd.concat([st.session_state.feedback_log, feedback], ignore_index=True)
    st.session_state.feedback_log.to_csv('feedback_log.csv', index=False)

def get_llm_response(prompt, detected_pest=None):
    """Get response from fine-tuned Azure OpenAI model"""
    try:
        user_content = prompt
        if detected_pest:
            risk = RISK_SCORES.get(detected_pest, 0.5)
            user_content = f"[Detected pest: {detected_pest}, Risk Score: {risk:.2f}] {prompt}"
        
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            max_tokens=600,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Azure OpenAI Error: {str(e)}\n\nCheck your .env file has AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY set correctly."

# Main UI
st.title("🐛 Agricultural Pest Detection with RLHF + LLM")
st.markdown("### Canadian Pest Management & Risk Assessment System with Adaptive Learning")

tab1, tab2, tab3 = st.tabs(["🔍 Detection", "📈 Analytics", "💬 Pest Expert Chat"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader("Choose a pest image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            # Reset correction flag if new image uploaded
            if st.session_state.last_uploaded_file != uploaded_file.name:
                st.session_state.pest_corrected = False
                st.session_state.last_uploaded_file = uploaded_file.name
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        if uploaded_file:
            st.subheader("🔍 Prediction Results")
            session_b0, session_b4 = load_models()
            if session_b0 and session_b4:
                ensemble, output_b0, output_b4, disagreement, w_b0, w_b4 = run_inference(image, session_b0, session_b4)
                pred_idx = np.argmax(ensemble)
                predicted_pest = PEST_CLASSES[pred_idx]
                confidence = float(ensemble[pred_idx])
                top_5_idx = np.argsort(ensemble)[-5:][::-1]
                top_5_classes = [PEST_CLASSES[i] for i in top_5_idx]
                # Only update current_detection if not corrected
                if not st.session_state.pest_corrected:
                    st.session_state.current_detection = predicted_pest
                st.success(f"**Detected: {predicted_pest.upper()}**")
                st.info(f"**Confidence: {confidence:.1%}**")
                st.caption(f"Ensemble Weights: B0={w_b0:.1%}, B4={w_b4:.1%}")
                threshold = st.session_state.thresholds[predicted_pest]
                needs_review = confidence < threshold
                if needs_review:
                    st.warning(f"⚠️ NEEDS HUMAN REVIEW (threshold: {threshold:.1%})")
                    risk_score = RISK_SCORES[predicted_pest]
                    risk_level = "High" if risk_score > 0.7 else "Medium" if risk_score > 0.4 else "Low"
                    st.error(f"Risk Level: {risk_level} (Score: {risk_score:.2f})")
                    st.markdown("---")
                    st.markdown("### Provide Feedback")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("✅ Correct", use_container_width=True, key="correct"):
                            old_t, new_t, msg = update_threshold_smart(predicted_pest, confidence, True, predicted_pest, top_5_classes)
                            new_w_b0, new_w_b4 = update_ensemble_weights(predicted_pest, predicted_pest, output_b0, output_b4)
                            log_feedback(predicted_pest, predicted_pest, confidence, True, True, old_t, new_t, new_w_b0, new_w_b4)
                            st.success(f"{msg}")
                            st.info(f"Threshold: {old_t:.2%} → {new_t:.2%}")
                            st.info(f"Weights: B0={new_w_b0:.1%}, B4={new_w_b4:.1%}")
                            st.balloons()
                    with col_b:
                        if st.button("❌ Wrong", use_container_width=True, key="wrong"):
                            st.session_state.show_correction = True
                    if st.session_state.get('show_correction', False):
                        st.markdown("#### Select Correct Pest Class:")
                        correct_class = st.selectbox("What is the correct pest?", PEST_CLASSES, key="correct_class_select")
                        if st.button("Submit Correction", key="submit_correction"):
                            in_top_5 = correct_class in top_5_classes
                            old_t, new_t, msg = update_threshold_smart(predicted_pest, confidence, False, correct_class, top_5_classes)
                            new_w_b0, new_w_b4 = update_ensemble_weights(predicted_pest, correct_class, output_b0, output_b4)
                            log_feedback(predicted_pest, correct_class, confidence, False, in_top_5, old_t, new_t, new_w_b0, new_w_b4)
                            st.error(f"{msg}")
                            st.info(f"Correction: {predicted_pest} → {correct_class}")
                            st.info(f"Threshold: {old_t:.2%} → {new_t:.2%}")
                            st.info(f"Weights: B0={new_w_b0:.1%}, B4={new_w_b4:.1%}")
                            if in_top_5:
                                st.success("✅ Correct answer was in Top 5!")
                            # Update current detection to corrected class
                            st.session_state.current_detection = correct_class
                            st.session_state.pest_corrected = True
                            st.session_state.show_correction = False
                            st.rerun()
                else:
                    st.success("✅ HIGH CONFIDENCE - Auto-approved")
                    risk_score = RISK_SCORES[predicted_pest]
                    risk_level = "High" if risk_score > 0.7 else "Medium" if risk_score > 0.4 else "Low"
                    if risk_level == "High":
                        st.error(f"⚠️ HIGH RISK - Loan requires pest management plan")
                    elif risk_level == "Medium":
                        st.warning(f"Medium Risk - Monitoring required")
                    else:
                        st.success(f"Low Risk - Standard loan processing")
                with st.expander("📊 Model Details"):
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric("Model Agreement", f"{(1-disagreement)*100:.1f}%")
                    with col_m2:
                        st.metric("B0 Confidence", f"{output_b0[pred_idx]:.1%}")
                    with col_m3:
                        st.metric("B4 Confidence", f"{output_b4[pred_idx]:.1%}")
                    st.subheader("Top 5 Predictions")
                    for idx in top_5_idx:
                        pest = PEST_CLASSES[idx]
                        prob = min(max(float(ensemble[idx]), 0.0), 1.0)
                        st.progress(prob, text=f"{pest}: {prob:.1%}")

with tab2:
    st.subheader("📊 Adaptive Learning Analytics")
    if len(st.session_state.feedback_log) > 0:
        metric_type = st.selectbox("Select Analysis Type", ["Adaptive Learning Overview", "Threshold Evolution", "Ensemble Weight Evolution", "Accuracy by Pest", "Top-5 Performance", "Model Performance Over Time"])
        if metric_type == "Adaptive Learning Overview":
            col1, col2, col3, col4 = st.columns(4)
            total = len(st.session_state.feedback_log)
            correct = st.session_state.feedback_log['was_correct'].sum()
            top5_correct = st.session_state.feedback_log['in_top_5'].sum()
            with col1:
                st.metric("Total Reviews", total)
            with col2:
                st.metric("Exact Match Accuracy", f"{(correct/total*100):.1f}%")
            with col3:
                st.metric("Top-5 Accuracy", f"{(top5_correct/total*100):.1f}%")
            with col4:
                st.metric("Avg Confidence", f"{st.session_state.feedback_log['confidence'].mean():.1%}")
            st.markdown("---")
            col_w1, col_w2 = st.columns(2)
            with col_w1:
                st.metric("Current B0 Weight", f"{st.session_state.ensemble_weights['b0']:.1%}")
                b0_acc = st.session_state.model_accuracy['b0']['correct'] / max(st.session_state.model_accuracy['b0']['total'], 1)
                st.caption(f"B0 Accuracy: {b0_acc:.1%}")
            with col_w2:
                st.metric("Current B4 Weight", f"{st.session_state.ensemble_weights['b4']:.1%}")
                b4_acc = st.session_state.model_accuracy['b4']['correct'] / max(st.session_state.model_accuracy['b4']['total'], 1)
                st.caption(f"B4 Accuracy: {b4_acc:.1%}")
        elif metric_type == "Threshold Evolution":
            fig = go.Figure()
            for pest in PEST_CLASSES:
                pest_data = st.session_state.feedback_log[st.session_state.feedback_log['predicted_pest'] == pest]
                if len(pest_data) > 0:
                    fig.add_trace(go.Scatter(x=pd.to_datetime(pest_data['timestamp']), y=pest_data['new_threshold'], mode='lines+markers', name=pest))
            fig.update_layout(title="Adaptive Threshold Evolution Over Time", xaxis_title="Time", yaxis_title="Confidence Threshold", height=500)
            st.plotly_chart(fig, use_container_width=True)
        elif metric_type == "Ensemble Weight Evolution":
            if len(st.session_state.weight_history) > 0:
                df_weights = pd.DataFrame(st.session_state.weight_history)
                df_weights['timestamp'] = pd.to_datetime(df_weights['timestamp'])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_weights['timestamp'], y=df_weights['b0_weight'], mode='lines+markers', name='EfficientNet-B0', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=df_weights['timestamp'], y=df_weights['b4_weight'], mode='lines+markers', name='EfficientNet-B4', line=dict(color='red')))
                fig.update_layout(title="Ensemble Weight Adaptation Over Time", xaxis_title="Time", yaxis_title="Model Weight", height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No weight history yet. Make predictions to see weight evolution.")
        elif metric_type == "Top-5 Performance":
            col1, col2 = st.columns(2)
            with col1:
                exact_match = st.session_state.feedback_log['was_correct'].sum()
                top5_match = st.session_state.feedback_log['in_top_5'].sum()
                total = len(st.session_state.feedback_log)
                df_perf = pd.DataFrame({'Metric': ['Exact Match', 'In Top-5', 'Not in Top-5'], 'Count': [exact_match, top5_match - exact_match, total - top5_match]})
                fig = px.pie(df_perf, values='Count', names='Metric', title='Prediction Performance Distribution')
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.metric("Learning Progress", f"{(top5_match/total*100):.1f}%", f"+{((top5_match/total) - (exact_match/total))*100:.1f}% from top-5")
                st.markdown("### Top-5 Benefit")
                st.info(f"**Exact Matches**: {exact_match}/{total}\n**Top-5 Matches**: {top5_match}/{total}\n**Improvement**: {top5_match - exact_match} additional correct")
        elif metric_type == "Accuracy by Pest":
            accuracy_data = []
            for pest in PEST_CLASSES:
                pest_feedback = st.session_state.feedback_log[st.session_state.feedback_log['predicted_pest'] == pest]
                if len(pest_feedback) > 0:
                    accuracy_data.append({'Pest': pest, 'Accuracy': pest_feedback['was_correct'].mean() * 100})
            if accuracy_data:
                fig = px.bar(pd.DataFrame(accuracy_data), x='Pest', y='Accuracy', title='Accuracy by Pest Class', color='Accuracy', color_continuous_scale='RdYlGn')
                st.plotly_chart(fig, use_container_width=True)
        elif metric_type == "Model Performance Over Time":
            df_time = st.session_state.feedback_log.copy()
            df_time['timestamp'] = pd.to_datetime(df_time['timestamp'])
            df_time = df_time.sort_values('timestamp')
            df_time['rolling_accuracy'] = df_time['was_correct'].rolling(window=min(10, len(df_time)), min_periods=1).mean() * 100
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_time['timestamp'], y=df_time['rolling_accuracy'], mode='lines', name='Rolling Accuracy', line=dict(color='blue', width=2)))
            fig.update_layout(title='System Accuracy Over Time (10-sample rolling average)', xaxis_title='Time', yaxis_title='Accuracy (%)', height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No feedback data yet. Start making predictions and providing feedback to see adaptive learning in action!")

with tab3:
    st.subheader("💬 Canadian Pest Management Expert")
    st.caption("Powered by fine-tuned GPT-4.1-nano with CFIA, CCFM, GFO, and Agriculture Canada knowledge")
    if st.session_state.current_detection:
        st.info(f"📍 Current detected pest: **{st.session_state.current_detection}**")
    st.markdown("### Ask about pest management, treatments, climate impacts, or loan risk")
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.chat_message("user").write(message['content'])
        else:
            st.chat_message("assistant").write(message['content'])
    if prompt := st.chat_input("Ask about pest management..."):
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        st.chat_message("user").write(prompt)
        response = get_llm_response(prompt, st.session_state.current_detection)
        st.session_state.chat_history.append({'role': 'assistant', 'content': response})
        st.chat_message("assistant").write(response)
    st.markdown("### Quick Questions:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🌱 Treatment Options", use_container_width=True):
            prompt = f"What are the treatment options for {st.session_state.current_detection}?" if st.session_state.current_detection else "What are general pest treatment options?"
            response = get_llm_response(prompt, st.session_state.current_detection)
            st.session_state.chat_history.append({'role': 'user', 'content': prompt})
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})
            st.rerun()
    with col2:
        if st.button("🏦 Loan Risk Assessment", use_container_width=True):
            prompt = f"What is the loan risk for {st.session_state.current_detection} infestation?" if st.session_state.current_detection else "How do pests affect agricultural loan risk?"
            response = get_llm_response(prompt, st.session_state.current_detection)
            st.session_state.chat_history.append({'role': 'user', 'content': prompt})
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})
            st.rerun()
    with st.expander("📚 Knowledge Base Sources"):
        st.markdown("""
        - [Agriculture Canada - Climate Change Impacts](https://agriculture.canada.ca/en/environment/climate-change/climate-change-impacts-agriculture)
        - [CCFM National Forest Pest Strategy](https://www.ccfm.org/wp-content/uploads/2020/08/National-Forest-Pest-Strategy_Pest-Risk-Analysis-Framework_User%E2%80%99s-Guide_EN.pdf)
        - [GFO Crop Pests Guide](https://gfo.ca/wp-content/uploads/2018/01/CropPests-reduced.pdf)
        - [CFIA Pest Evaluation](https://inspection.canada.ca/en/plant-health/horticulture/how-we-evaluate)
        """)

with st.sidebar:
    st.header("📊 Adaptive System Status")
    total_feedback = len(st.session_state.feedback_log)
    if total_feedback > 0:
        accuracy = (st.session_state.feedback_log['was_correct'].sum() / total_feedback) * 100
        top5_acc = (st.session_state.feedback_log['in_top_5'].sum() / total_feedback) * 100
        st.metric("Exact Match Accuracy", f"{accuracy:.1f}%")
        st.metric("Top-5 Accuracy", f"{top5_acc:.1f}%")
    st.metric("Total Reviews", total_feedback)
    st.markdown("---")
    st.subheader("Ensemble Weights")
    st.progress(st.session_state.ensemble_weights['b0'], text=f"B0: {st.session_state.ensemble_weights['b0']:.1%}")
    st.progress(st.session_state.ensemble_weights['b4'], text=f"B4: {st.session_state.ensemble_weights['b4']:.1%}")
    st.markdown("---")
    st.subheader("Current Thresholds")
    for pest in PEST_CLASSES:
        threshold = st.session_state.thresholds[pest]
        risk = RISK_SCORES[pest]
        color = "🔴" if risk > 0.7 else "🟡" if risk > 0.4 else "🟢"
        st.text(f"{color} {pest}: {threshold:.2%}")
    st.markdown("---")
    st.subheader("📥 Export Data")
    st.download_button("Download Thresholds", data=json.dumps(st.session_state.thresholds, indent=2), file_name="learned_thresholds.json", mime="application/json")
    if len(st.session_state.feedback_log) > 0:
        st.download_button("Download Feedback Log", data=st.session_state.feedback_log.to_csv(index=False), file_name="feedback_log.csv", mime="text/csv")

st.markdown("---")
st.caption("🏦 Adaptive Pest Detection System - EfficientNet Ensemble + RLHF + GPT-4.1-nano Fine-tuned")