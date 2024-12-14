# Cuisine Fusion Project

This repository provides a complete implementation of a system designed to explore culinary transformations across different cuisines. By leveraging clustering techniques, pre-trained ingredient embeddings, and machine learning models, this project enables ingredient substitutions and recipe transformations between regional cuisines.

---

## Project Structure

The project consists of the following two main files:

1. **`cuisine_fusion_documented.ipynb`:**
   - A Jupyter Notebook that preprocesses recipe datasets, clusters ingredients based on their regional cuisine, and calculates transformations using cosine similarity and pre-trained embeddings.
   - It provides step-by-step insights into clustering recipes, visualizing transformations, and performing regional recipe adaptations.

2. **`inference.py`:**
   - A Python script for real-time inference using command-line arguments.
   - It takes a source cuisine, a destination cuisine, and a list of ingredients as input.
   - Outputs the transformed ingredients, the number of substitutions, and transformation statistics.

---

## Features

1. **Data Preprocessing:**
   - Standardizes ingredient names for uniformity.
   - Clusters recipes based on regional cuisines.

2. **Ingredient Embedding:**
   - Uses pre-trained embeddings for ingredient representation.
   - Computes cosine similarity to identify substitutions.

3. **Recipe Transformation:**
   - Implements K-Nearest Neighbors (KNN) for clustering and regional transformation.
   - Calculates thresholds for ingredient substitutions to adapt recipes between cuisines.

4. **Visualization:**
   - Heatmaps showcasing the success of recipe transformations.
   - Bar plots indicating the complexity of transformations across cuisines.

5. **Real-time Inference:**
   - Command-line support to analyze and adapt recipes in real-time.

---

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/cuisine-fusion.git
   cd cuisine-fusion
   ```

2. **Install Dependencies:**
   Make sure you have Python 3.x installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
   Required libraries include:
   - `pandas`
   - `numpy`
   - `torch`
   - `matplotlib`
   - `argparse`

3. **Prepare Dataset:**
   - Place the dataset files (e.g., `RecipeDB_Merged.csv`) and embedding files (e.g., `ingredient_embeddings.txt` and `final_ingredient_embeddings.pt`) in the appropriate directory as specified in the code.

---

## Usage

### 1. Running the Jupyter Notebook
Open `cuisine_fusion_documented.ipynb` in Jupyter Notebook and execute the cells sequentially to:
   - Preprocess and cluster recipes.
   - Visualize transformation success rates.
   - Analyze regional culinary differences.

### 2. Using the Command-line Script
The script `inference.py` allows for real-time ingredient transformation:
   ```bash
   python inference.py -s "Indian Subcontinent" -d "Mexican" -i "['onion', 'garlic', 'tomato']"
   ```
   **Example Output:**
   ```
   Source String: Indian Subcontinent
   Destination String: Mexican
   Item List: ['onion', 'garlic', 'tomato']
   Converted Item List: ['cebolla', 'ajo', 'jitomate']
   Is Converted: True
   No. of new and modified ingredients added: 2
   No. of unaffected items: 1
   Original list length: 3
   Final list length: 5
   ```

---

## File Details

### 1. `cuisine_fusion_documented.ipynb`
   - **Key Sections:**
     1. **Data Loading and Preprocessing:**
        - Cleans and organizes recipe data for clustering.
     2. **Clustering Recipes:**
        - Uses tuple representation to group similar recipes.
     3. **Embedding and Similarity Computations:**
        - Applies pre-trained embeddings for cosine similarity calculations.
     4. **Visualization:**
        - Heatmaps and bar plots for transformation success rates and complexity.

### 2. `inference.py`
   - **Key Functions:**
     1. **`prep_ing(x)`:**
        - Preprocesses ingredient names (e.g., replacing spaces with underscores).
     2. **`get_tuple_embedding(tup_name)`:**
        - Computes an average embedding for a tuple of ingredients.
     3. **`check_transformation(model, old_ing, new_ing)`:**
        - Checks the validity of a transformation using an MLP model.
     4. **`get_threshold(ingredient_list, tuple_list, src, dest)`:**
        - Calculates the number of substitutions required for a recipe transformation.

---

## Visualization Examples

### Heatmap: Transformation Success Rate
Displays the success rates of recipe transformations between different regions. For instance, Indian to Mexican transformations have a high success rate, while Canadian transformations might require more substitutions.

### Bar Plot: Ingredient Changes
Illustrates the complexity of transformations by showing the number of ingredient changes for each regional pair.

---

## Future Improvements

1. Enhance the embedding model to incorporate additional contextual data (e.g., cooking methods or ingredient categories).
2. Improve visualization by adding dynamic interactivity (e.g., tooltips or sliders).
3. Expand the dataset to include more cuisines and regional specialties.

---

## Contributions

Feel free to open issues or submit pull requests to improve this project. We welcome any feedback or suggestions!

---
