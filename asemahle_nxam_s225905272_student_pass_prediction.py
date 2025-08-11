import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class StudentPassPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = ['study_hours', 'sleep_hours']
        self.bias = None
        self.weights = None
        
    def generate_synthetic_data(self, n_samples=1000):
        """
        Generate synthetic data for student study patterns
        Study hours: 0-16 hours per day
        Sleep hours: 4-12 hours per day
        """
        print("🎓 Generating synthetic student data...")
        
        # Generate study hours (more realistic distribution)
        study_hours = np.random.normal(6, 2, n_samples)
        study_hours = np.clip(study_hours, 0, 16)
        
        # Generate sleep hours (more realistic distribution)
        sleep_hours = np.random.normal(7.5, 1.5, n_samples)
        sleep_hours = np.clip(sleep_hours, 4, 12)
        
        # Create features DataFrame
        features = pd.DataFrame({
            'study_hours': study_hours,
            'sleep_hours': sleep_hours
        })
        
        # Create labels based on realistic patterns
        # Higher study hours and moderate sleep hours increase pass probability
        pass_probability = (
            0.3 * (features['study_hours'] / 16) +  # Study hours contribution
            0.2 * (1 - abs(features['sleep_hours'] - 7.5) / 4) +  # Sleep optimization
            0.1 * np.random.random(n_samples)  # Random noise
        )
        
        # Convert to binary labels (pass/fail)
        labels = (pass_probability > 0.5).astype(int)
        
        # Create final dataset
        self.data = pd.concat([features, pd.Series(labels, name='pass')], axis=1)
        
        print(f"✅ Generated {n_samples} student records")
        print(f"📊 Pass rate: {labels.mean():.1%}")
        print(f"📚 Study hours range: {features['study_hours'].min():.1f} - {features['study_hours'].max():.1f}")
        print(f"😴 Sleep hours range: {features['sleep_hours'].min():.1f} - {features['sleep_hours'].max():.1f}")
        
        return self.data
    
    def prepare_data(self):
        """Prepare data for training"""
        print("\n🔧 Preparing data for training...")
        
        # Separate features and labels
        X = self.data[self.feature_names]
        y = self.data['pass']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_train_scaled = X_train_scaled
        self.X_test_scaled = X_test_scaled
        
        print(f"✅ Training set: {X_train.shape[0]} samples")
        print(f"✅ Test set: {X_test.shape[0]} samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self):
        """Train the logistic regression model"""
        print("\n🚀 Training logistic regression model...")
        
        # Create and train model
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(self.X_train_scaled, self.y_train)
        
        # Extract weights and bias
        self.weights = self.model.coef_[0]
        self.bias = self.model.intercept_[0]
        
        print("✅ Model training completed!")
        print(f"📊 Model bias (intercept): {self.bias:.4f}")
        print(f"⚖️ Feature weights:")
        for feature, weight in zip(self.feature_names, self.weights):
            print(f"   {feature}: {weight:.4f}")
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n📈 Evaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test_scaled)
        y_pred_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"✅ Model Accuracy: {accuracy:.1%}")
        print("\n📊 Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Fail', 'Pass']))
        
        # Store predictions for analysis
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        
        return accuracy
    
    def analyze_bias_and_weights(self):
        """Analyze model bias and feature importance"""
        print("\n🔍 Analyzing model bias and feature importance...")
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Weight': self.weights,
            'Absolute_Weight': np.abs(self.weights)
        }).sort_values('Absolute_Weight', ascending=False)
        
        print("\n📊 Feature Importance (by absolute weight):")
        print(feature_importance)
        
        # Bias analysis
        print(f"\n🎯 Model Bias Analysis:")
        print(f"   Intercept (bias): {self.bias:.4f}")
        
        if self.bias > 0:
            print("   📈 Positive bias: Model tends to predict 'Pass' more often")
        else:
            print("   📉 Negative bias: Model tends to predict 'Fail' more often")
        
        # Decision boundary analysis
        print(f"\n🎯 Decision Boundary Analysis:")
        print(f"   P(Pass) = 1 / (1 + e^(-({self.bias:.4f} + {self.weights[0]:.4f}*study_hours + {self.weights[1]:.4f}*sleep_hours)))")
        
        return feature_importance
    
    def predict_sample(self, study_hours, sleep_hours):
        """Predict pass/fail for a specific student"""
        # Create sample
        sample = np.array([[study_hours, sleep_hours]])
        
        # Scale the sample
        sample_scaled = self.scaler.transform(sample)
        
        # Make prediction
        prediction = self.model.predict(sample_scaled)[0]
        probability = self.model.predict_proba(sample_scaled)[0][1]
        
        result = "PASS" if prediction == 1 else "FAIL"
        
        print(f"\n🎓 Student Prediction:")
        print(f"   Study Hours: {study_hours}")
        print(f"   Sleep Hours: {sleep_hours}")
        print(f"   Prediction: {result}")
        print(f"   Pass Probability: {probability:.1%}")
        
        return prediction, probability
    
    def visualize_results(self):
        """Create comprehensive visualizations"""
        print("\n🎨 Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Student Pass/Fail Prediction Analysis', fontsize=16, fontweight='bold')
        
        # 1. Data Distribution
        axes[0, 0].hist(self.data['study_hours'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Study Hours Distribution')
        axes[0, 0].set_xlabel('Study Hours')
        axes[0, 0].set_ylabel('Frequency')
        
        axes[0, 1].hist(self.data['sleep_hours'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Sleep Hours Distribution')
        axes[0, 1].set_xlabel('Sleep Hours')
        axes[0, 1].set_ylabel('Frequency')
        
        # 2. Pass/Fail by Study Hours
        pass_study = self.data[self.data['pass'] == 1]['study_hours']
        fail_study = self.data[self.data['pass'] == 0]['study_hours']
        
        axes[0, 2].hist([pass_study, fail_study], bins=15, alpha=0.7, 
                        label=['Pass', 'Fail'], color=['green', 'red'])
        axes[0, 2].set_title('Study Hours: Pass vs Fail')
        axes[0, 2].set_xlabel('Study Hours')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        
        # 3. Scatter plot with decision boundary
        scatter = axes[1, 0].scatter(self.data['study_hours'], self.data['sleep_hours'], 
                                    c=self.data['pass'], cmap='RdYlGn', alpha=0.6)
        axes[1, 0].set_title('Study vs Sleep Hours (Pass/Fail)')
        axes[1, 0].set_xlabel('Study Hours')
        axes[1, 0].set_ylabel('Sleep Hours')
        axes[1, 0].legend(handles=scatter.legend_elements()[0], labels=['Fail', 'Pass'])
        
        # 4. Feature importance
        feature_names = self.feature_names
        weights = self.weights
        colors = ['skyblue', 'lightgreen']
        
        bars = axes[1, 1].bar(feature_names, weights, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Feature Weights')
        axes[1, 1].set_ylabel('Weight Value')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels on bars
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{weight:.3f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # 5. Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 2],
                   xticklabels=['Fail', 'Pass'], yticklabels=['Fail', 'Pass'])
        axes[1, 2].set_title('Confusion Matrix')
        axes[1, 2].set_xlabel('Predicted')
        axes[1, 2].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
        
        # Additional detailed analysis
        self._create_detailed_analysis()
    
    def _create_detailed_analysis(self):
        """Create additional detailed analysis plots"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 1. Probability distribution
        axes[0].hist(self.y_pred_proba[self.y_test == 0], bins=20, alpha=0.7, 
                    label='Fail', color='red', density=True)
        axes[0].hist(self.y_pred_proba[self.y_test == 1], bins=20, alpha=0.7, 
                    label='Pass', color='green', density=True)
        axes[0].set_title('Prediction Probability Distribution')
        axes[0].set_xlabel('Pass Probability')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[0].axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Decision Threshold')
        axes[0].legend()
        
        # 2. Study hours vs Pass probability
        study_range = np.linspace(0, 16, 100)
        sleep_avg = self.data['sleep_hours'].mean()
        
        # Create samples for different study hours
        samples = np.column_stack([study_range, np.full_like(study_range, sleep_avg)])
        samples_scaled = self.scaler.transform(samples)
        probabilities = self.model.predict_proba(samples_scaled)[:, 1]
        
        axes[1].plot(study_range, probabilities, 'b-', linewidth=2, label='Pass Probability')
        axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Threshold')
        axes[1].set_title(f'Pass Probability vs Study Hours (Sleep: {sleep_avg:.1f}h)')
        axes[1].set_xlabel('Study Hours')
        axes[1].set_ylabel('Pass Probability')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """Generate a comprehensive report"""
        print("\n" + "="*60)
        print("🎓 STUDENT PASS/FAIL PREDICTION MODEL REPORT")
        print("="*60)
        
        print(f"\n📊 DATASET SUMMARY:")
        print(f"   Total students: {len(self.data)}")
        print(f"   Pass rate: {self.data['pass'].mean():.1%}")
        print(f"   Fail rate: {1 - self.data['pass'].mean():.1%}")
        
        print(f"\n🔧 MODEL DETAILS:")
        print(f"   Algorithm: Logistic Regression")
        print(f"   Features: {', '.join(self.feature_names)}")
        print(f"   Bias (intercept): {self.bias:.4f}")
        
        print(f"\n⚖️ FEATURE WEIGHTS:")
        for feature, weight in zip(self.feature_names, self.weights):
            impact = "Positive" if weight > 0 else "Negative"
            print(f"   {feature}: {weight:.4f} ({impact} impact)")
        
        print(f"\n🎯 INTERPRETATION:")
        if self.weights[0] > 0:
            print(f"   📚 More study hours increase pass probability")
        else:
            print(f"   📚 More study hours decrease pass probability")
            
        if self.weights[1] > 0:
            print(f"   😴 More sleep hours increase pass probability")
        else:
            print(f"   😴 More sleep hours decrease pass probability")
        
        print(f"\n📈 MODEL PERFORMANCE:")
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f"   Accuracy: {accuracy:.1%}")
        
        # Calculate additional metrics
        cm = confusion_matrix(self.y_test, self.y_pred)
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
        
        print(f"   Precision: {precision:.1%}")
        print(f"   Recall: {recall:.1%}")
        
        print("\n" + "="*60)

def main():
    """Main function to run the complete analysis"""
    print("🎓 Student Pass/Fail Prediction Model")
    print("=" * 50)
    
    # Initialize the predictor
    predictor = StudentPassPredictor()
    
    # Generate data
    data = predictor.generate_synthetic_data(n_samples=1000)
    
    # Prepare data
    predictor.prepare_data()
    
    # Train model
    model = predictor.train_model()
    
    # Evaluate model
    accuracy = predictor.evaluate_model()
    
    # Analyze bias and weights
    feature_importance = predictor.analyze_bias_and_weights()
    
    # Generate report
    predictor.generate_report()
    
    # Make sample predictions
    print("\n🎯 SAMPLE PREDICTIONS:")
    predictor.predict_sample(study_hours=8, sleep_hours=7)
    predictor.predict_sample(study_hours=2, sleep_hours=5)
    predictor.predict_sample(study_hours=12, sleep_hours=8)
    
    # Create visualizations
    predictor.visualize_results()
    
    print("\n✅ Analysis complete! Check the visualizations above.")
    
    return predictor

if __name__ == "__main__":
    # Run the complete analysis
    predictor = main() 