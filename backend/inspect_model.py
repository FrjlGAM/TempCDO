"""
Script to inspect the loaded ML model and understand its structure.
This helps determine what features the model expects as input.
"""
import joblib
from pathlib import Path
import numpy as np

MODEL_PATH = Path(__file__).parent.parent / "temp_forecaster_model.joblib"

def inspect_model():
    """Load and inspect the model to understand its structure"""
    try:
        print(f"Loading model from: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        
        print("\n" + "="*60)
        print("MODEL INSPECTION")
        print("="*60)
        
        # Get model type
        print(f"\nModel Type: {type(model).__name__}")
        print(f"Model Class: {type(model)}")
        
        # Check if it's a scikit-learn model
        if hasattr(model, 'predict'):
            print("\n✓ Model has 'predict' method")
        else:
            print("\n✗ Model does not have 'predict' method")
        
        # Check for feature names or feature requirements
        if hasattr(model, 'feature_names_in_'):
            print(f"\nFeature Names: {model.feature_names_in_}")
            print(f"Number of Features Expected: {len(model.feature_names_in_)}")
        elif hasattr(model, 'n_features_in_'):
            print(f"\nNumber of Features Expected: {model.n_features_in_}")
        else:
            print("\n⚠ Could not determine feature requirements from model attributes")
        
        # Check for other useful attributes
        print("\nModel Attributes:")
        attrs = [attr for attr in dir(model) if not attr.startswith('_')]
        important_attrs = ['n_features_in_', 'feature_names_in_', 'coef_', 'intercept_', 
                          'n_estimators', 'max_depth', 'n_components', 'components_']
        for attr in important_attrs:
            if hasattr(model, attr):
                value = getattr(model, attr)
                if isinstance(value, np.ndarray):
                    print(f"  - {attr}: array shape {value.shape}")
                else:
                    print(f"  - {attr}: {value}")
        
        # Try to get model parameters
        if hasattr(model, 'get_params'):
            print("\nModel Parameters:")
            params = model.get_params()
            for key, value in list(params.items())[:10]:  # Show first 10
                print(f"  - {key}: {value}")
            if len(params) > 10:
                print(f"  ... and {len(params) - 10} more parameters")
        
        # Test prediction with sample data
        print("\n" + "="*60)
        print("TESTING PREDICTION")
        print("="*60)
        
        # Try different feature array shapes
        test_shapes = [
            (1, 1),   # Single feature
            (1, 2),   # Two features
            (1, 6),   # Six features (date components)
            (1, 12),  # Twelve features (with cyclical)
        ]
        
        for shape in test_shapes:
            try:
                test_features = np.random.rand(*shape)
                prediction = model.predict(test_features)
                print(f"\n✓ Prediction successful with shape {shape}")
                print(f"  Input shape: {test_features.shape}")
                print(f"  Output shape: {prediction.shape if hasattr(prediction, 'shape') else 'scalar'}")
                print(f"  Sample prediction: {prediction[0] if hasattr(prediction, '__iter__') else prediction}")
                break  # Stop at first successful shape
            except Exception as e:
                print(f"\n✗ Prediction failed with shape {shape}: {str(e)}")
        
        # If it's a pipeline, inspect steps
        if hasattr(model, 'steps') or hasattr(model, 'named_steps'):
            print("\n" + "="*60)
            print("PIPELINE INSPECTION")
            print("="*60)
            if hasattr(model, 'steps'):
                print(f"Pipeline steps: {model.steps}")
            if hasattr(model, 'named_steps'):
                print(f"Named steps: {list(model.named_steps.keys())}")
        
        print("\n" + "="*60)
        print("INSPECTION COMPLETE")
        print("="*60)
        
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
    except Exception as e:
        print(f"Error inspecting model: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_model()

