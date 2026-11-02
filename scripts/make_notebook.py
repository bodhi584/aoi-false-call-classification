import json
import os

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Understanding - Siemens AOI Project\n",
    "This notebook provides an exploratory data analysis and visual summary of the Siemens AOI dataset for Issue #35."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.image as mpimg\n",
    "import os\n",
    "\n",
    "def show_img(path, title):\n",
    "    if os.path.exists(path):\n",
    "        img = mpimg.imread(path)\n",
    "        plt.figure(figsize=(10, 6))\n",
    "        plt.imshow(img)\n",
    "        plt.axis('off')\n",
    "        plt.title(title, fontsize=14)\n",
    "        plt.show()\n",
    "    else:\n",
    "        print(f'Image not found at {path}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Share of False Calls per Inspection Type\n",
    "**Description:** Shows the percentage of false alarms relative to total inspections per inspection type."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "show_img('../docs/img/data_false_calls_per_inspection_type.png', 'Share of False Calls per Inspection Type')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Flagged Boards and True Defects per Week\n",
    "**Description:** Compares the weekly volume of total flagged AOI boards against confirmed true defects over time."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "show_img('../docs/img/data_flagged_boards_and_true_defects_per_week.png', 'Flagged Boards and True Defects per Week')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Total Inspections per Inspection Type (Top 10)\n",
    "**Description:** Highlights the top 10 inspection types with the highest total inspection volume in the dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "show_img('../docs/img/data_false_calls_per_component_type.png', 'Total Inspections per Inspection Type (Top 10)')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

os.makedirs("notebooks", exist_ok=True)
nb_path = os.path.join("notebooks", "data_understanding.ipynb")
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook_content, f, indent=2)

print(f"Successfully generated notebook at: {nb_path}")