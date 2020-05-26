# VAE-morpher
This repository contains a set of scripts that I produced during my masters thesis in Cyber Security for the University of Twente.

Since this was all personal code, no documentation is present except for a few comments in the files. However, if you have any questions regarding the scripts or need some help getting it working, feel free to contact me at `vae-morpher(at)rienheuver[dot]nl`.

## Usage
This repository comes with a trained model: `model.pt`.

### Make a morph
If you want to make a morph, simply use `morpher.py [file1] [file2] (outputFile)` where the outputFile is optional.

### Train the model
For this you need `cnn-vae.py`.

### Experiment
I used `recognize.py` for a plethora of different experiments during my research, which is why there is so much code commented. It could take some experimentation on your side to make it do what you want.
