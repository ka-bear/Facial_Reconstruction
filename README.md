# Run Instructions
## Data Generation
1. Follow instructions from https://github.com/cheind/pytorch-blender to setup blendtorch
2. Download FLAME and AlbedoMM models and paste under flame/model
3. Change output directories and run generate_faces.py
4. Change directories and run postprocess.py

# Training/Evaluation
To run the training and evaluation code, ensure that the dataset directory is correct
(first line of the main function), and the model defined in the code is the same as the
model loaded by the checkpoint. From there the training and evaluation files should be able
to run on their own.


