import fiftyone as fo
import fiftyone.brain as fob
import os
import numpy as np
from collections import Counter

# dataset = fo.load_dataset("hotel8k")
# dataset.delete()
dataset = fo.Dataset("hotel8k") # Creating a new Dataset
train_dir = "/shared/data/hotel_8k_images/train"


for root, dirs, files in os.walk(train_dir): # Goes through the entire hotel_8k folder and extracts images
    for file_name in files:
        # Check if the file is a .jpg file
        if file_name.lower().endswith(".jpg"):
            file_path = os.path.join(root, file_name)
            label = file_path.split('/')[-2]
            sample = fo.Sample(filepath=file_path)
            sample["Label"] = label # Labeling images based on hotel
            dataset.add_sample(sample) # Adding sample images into the new Dataset

embeddings = np.load('/shared/data/hotel_8k_images/train.npy')  # Loading file with embedded features

for sample, embedding in zip(dataset, embeddings): # Assuming that the embedding order aligned with the sample order
    sample["custom_embeddings"] = embedding
    sample.save() # Saves embeddings into the Dataset

label_counts = Counter(dataset.values("Label")) # Count the number of samples per class

top_classes = [label for label, count in label_counts.most_common(20)] # Get the top 20 most frequent classes

subsampled_samples = []

for class_label in top_classes:
    class_samples = dataset.match({"Label": class_label}) # Extracting all the images for that specific class/hotel
    for sample in class_samples:
        sample["original_sample_id"] = sample.id
        subsampled_samples.append(sample)
    
    subsampled_samples.extend(class_samples) # Add all samples from this class into the subsampled list.


subsampled_dataset = fo.Dataset(name="hotel8k_subsampled") # Create new Dataset for subsamples
subsampled_dataset.add_samples(subsampled_samples)

# Computes the similarity scores for the subsampled Dataset
similarity_matrix = fob.compute_similarity(subsampled_dataset, num_workers=2, brain_key="similarity_index", embedding_field="custom_embeddings") 

sorted_view = subsampled_dataset.sort_by_similarity(
    query=subsampled_dataset.first().id, # Sorts the subsampled Dataset based on the first sample of the dataset
    brain_key="similarity_index"
)

visualization = fob.compute_visualization( # Create a visualization of the sorted subsampled Dataset
    sorted_view,
    brain_key="viz_sorted",
    embedding_field="custom_embeddings",
    method="umap",  
    num_dims=2
)

if __name__ == "__main__":
    session = fo.launch_app(sorted_view)
    session.wait()
