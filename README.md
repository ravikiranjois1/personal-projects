# Personal Projects

This is a repository for all the personal projects done by me
It includes the code, files used and the reports for the same

### A* Search:
Search the map of Mendon Ponds Park for the fastest path through given a set of points based on season and terrain information

Usage:
- $python3 lab1.py terrain.png <elevation_info> <coordinates_to_cover> <season> <output_image>.png

### English-Dutch Language Classifier:
Classify the lines in a file to English or Dutch using the Decision tree algorithm and also boosting the accuracy of the results using Adaboost algorithm

#### Usage:
The command line argument for Decision Tree:
Training: $python3 trainer.py train train_2500.txt hype_out_dt dt
Predict: $python3 trainer.py predict hype_out_dt test.dat

The command line argument for Adaboost: 
Training: $python3 trainer.py train train_2500.txt hype_out_ada ada 
Predict: $python3 trainer.py predict hype_out_ada test.dat
