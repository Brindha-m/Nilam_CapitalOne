<img width="200" height="200" alt="3147fb2d-77b0-4750-8951-ec3a92b44786" src="https://github.com/user-attachments/assets/fcc4a050-49c8-4cba-8ea7-6e76e5cfbb4c" />

<br>

**Nilam (Native Intelligence for Land and Agriculture Management) is an integrated AI-powered agricultural  multilingual chat assistance, leaf disease detection, and personalized crop recommendation systems.**

## Modules Architecture

<img width="1242" height="961" alt="NILAM ARCH" src="https://github.com/user-attachments/assets/52861f5c-95d3-41ff-b3b1-8e8b7279fbe4" />


### 🌱 **Nilam Chat**
- **AI-Powered Assistant**: Powered by Google's Generative AI (Gemini).
- **Specialized in farming queries, crop advice and available in multiple Indian languages adding on with market trends and price analysis**

### 🍃 **Leafine**
- **Leaf Disease Detection**: Custom trained on our leafine dataset with improved YOLOv8 architecture Resnext-50 with XAI.
- **Recognize & Perceive the leaf illness and figure out how to treat them!**

### 🧠 **NilamSense**
- **Smart Crop Recommendation**: AI-driven crop selection system custom trained data collected from OWM (Open weather map), Data.gov.in, copernicus trained with bidirectional stacked LSTM.
- **Recommend suitable crops for a specific location**: EDA combined data is @ data/nilamdata.csv

## 📋 Prerequisites

- Python 3.9 or higher
- Google Gemini API Key (for Nilam Chat)
- my .pkl file is more than 800mb lfs didn't support, so recommended to run simple_model.py locally (for Nilam Sense)

## 🛠️ Installation

1. **Clone the repository**
   
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API Keys**
   - Create a `.streamlit/secrets.toml` file
   - Add your Gemini API key:
   ```toml
   GEMINI_API_KEY = "your-gemini-api-key-here"
   ```

## 🚀 Usage

### Running the Application

**Start the main application**
   ```bash
   streamlit run main.py
   ```
