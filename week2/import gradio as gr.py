import gradio as gr
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Create or load a model
# For demonstration, let's create a simple model
X = np.random.rand(100, 4)
y = 5 * X[:, 0] + 2 * X[:, 1] - 3 * X[:, 2] + X[:, 3] + np.random.randn(100) * 0.5
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model
with open('simple_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Create prediction function
def predict(feature1, feature2, feature3, feature4):
    """Predict using the trained model"""
    # Prepare input features
    features = np.array([[feature1, feature2, feature3, feature4]])
    
    # Make prediction
    prediction = model.predict(features)[0]
    
    # Format the result
    return f"## Predicted value: {prediction:.2f}"

# Create Gradio interface
with gr.Blocks(title="Simple Predictor") as demo:
    gr.Markdown("# Feature Value Predictor")
    gr.Markdown("Enter values for features to get a prediction")
    
    with gr.Row():
        with gr.Column():
            feature1 = gr.Slider(minimum=0, maximum=1, value=0.5, label="Feature 1")
            feature2 = gr.Slider(minimum=0, maximum=1, value=0.5, label="Feature 2")
            feature3 = gr.Slider(minimum=0, maximum=1, value=0.5, label="Feature 3")
            feature4 = gr.Slider(minimum=0, maximum=1, value=0.5, label="Feature 4")
            
            predict_btn = gr.Button("Predict")
            
        with gr.Column():
            result = gr.Markdown("## Prediction will appear here")
    
    predict_btn.click(
        fn=predict,
        inputs=[feature1, feature2, feature3, feature4],
        outputs=result
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()