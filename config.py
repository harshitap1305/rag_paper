import os
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import google.generativeai as genai

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Set matplotlib backend and style
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Config
DB_DIR = "./chroma_db"
COLLECTION_NAME = "papers"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
N_RESULTS = 8
MAX_RESPONSE_LENGTH = 2000

# Gemini setup
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

# Crossref API for citations
CROSSREF_API = "https://api.crossref.org/works"

# Professional color palettes
PROFESSIONAL_COLORS = {
    'primary': '#2E86C1',
    'secondary': '#28B463', 
    'accent': '#F39C12',
    'warning': '#E74C3C',
    'info': '#8E44AD',
    'success': '#27AE60',
    'gradient': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe']
}