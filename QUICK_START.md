#  CineMatch Quick Start Guide

##  Get Started in 5 Minutes

### Option 1: Demo Mode (No Dataset Required)
```bash
# Run the demo to see the system in action
python demo.py
```

### Option 2: Full System (Requires Dataset)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download MovieLens 100k dataset
# Visit: https://grouplens.org/datasets/movielens/100k/
# Extract u.data, u.item, u.user to data/ folder

# 3. Run the main application
python main.py

# 4. Launch web interface
streamlit run app.py
```

##  What You'll See

### Demo Mode Features:
-  Sample data generation
-  Collaborative filtering demonstration
-  Content-based filtering demonstration
-  Hybrid system explanation
-  Evaluation metrics overview
-  Sample visualizations

### Full System Features:
-  Real MovieLens 100k dataset
-  User-based collaborative filtering
-  Content-based recommendations
-  Hybrid recommendation system
-  Comprehensive evaluation
-  Interactive web interface
-  Detailed analysis and visualizations

##  Interactive Features

### Web Application (Streamlit)
- **Home Page**: Dataset overview and statistics
- **Recommendations**: Get personalized movie recommendations
- **Data Analysis**: Explore the dataset with interactive charts
- **Movie Explorer**: Find similar movies and explore genres
- **System Settings**: Configure hybrid system weights

### Jupyter Notebook
- **Step-by-step analysis**: Follow the complete workflow
- **Interactive cells**: Run and modify code
- **Visualizations**: Create custom plots and charts
- **Experimentation**: Try different algorithms and parameters

##  System Requirements

### Minimum Requirements:
- Python 3.7+
- 4GB RAM
- 1GB free disk space

### Recommended:
- Python 3.8+
- 8GB RAM
- 2GB free disk space
- Modern web browser (for Streamlit app)

##  Expected Performance

### Demo Mode:
-  Runtime: ~30 seconds
-  Memory: ~100MB
-  Visualizations: 4 charts
-  Sample recommendations: 3 per method

### Full System:
-  Runtime: ~2-5 minutes (first run)
-  Memory: ~500MB
-  Visualizations: 10+ charts
-  Recommendations: 5-10 per method
-  Users: 943
-  Movies: 1,682
-  Ratings: 100,000

##  Learning Path

### Beginner:
1. Run `python demo.py` to understand concepts
2. Open `CineMatch.ipynb` and follow along
3. Try the Streamlit web app
4. Experiment with different parameters

### Intermediate:
1. Modify algorithms in the source code
2. Add new evaluation metrics
3. Try different hybrid approaches
4. Implement new similarity measures

### Advanced:
1. Add deep learning models
2. Implement real-time recommendations
3. Create API endpoints
4. Deploy to cloud platforms

##  Troubleshooting

### Common Issues:

**"Module not found" errors:**
```bash
pip install -r requirements.txt
```

**"Dataset not found" errors:**
- Download MovieLens 100k dataset
- Extract files to `data/` folder
- Or run demo mode instead

**Streamlit app won't start:**
```bash
pip install streamlit
streamlit run app.py
```

**Jupyter notebook issues:**
```bash
pip install jupyter
jupyter notebook CineMatch.ipynb
```

### Getting Help:
1. Check the `README.md` for detailed instructions
2. Review `PROJECT_SUMMARY.md` for technical details
3. Run `python test_system.py` to diagnose issues
4. Check the demo mode first to verify basic functionality

##  Success Indicators

You'll know everything is working when you see:

### Demo Mode:
-  "Sample dataset created" message
-  Collaborative filtering demonstration
-  Content-based filtering demonstration
-  Sample visualizations displayed
-  "DEMO COMPLETED!" message

### Full System:
-  "Data loaded successfully" message
-  User-movie matrix created
-  Recommendations generated for sample users
-  Evaluation metrics calculated
-  Streamlit app opens in browser

##  Next Steps

Once you have the system running:

1. **Explore the Code**: Read through the implementation files
2. **Experiment**: Try different parameters and algorithms
3. **Extend**: Add new features or evaluation metrics
4. **Deploy**: Consider deploying to cloud platforms
5. **Share**: Use as a portfolio project or learning resource

---

**Happy Recommending! ✨**

