# Airbnb

## Milestone 3
The milestone focusses on preparing and cleaning both the tabular and the image data from the given dataset.

For the tabular data I used the pandas python package to transfer the data from the csv file into a dataframe which can then easily be manipulated. Using the .dropna() pandas method I could easily remove the rows which had missing valus for the ratings columns or the decription, similarly using the .fillna() method I added default values for the columns which required them. Cleaning the description data was more complicated, since they were saved as a string in the form of a python list. These were then saved in a pandas series and used to replace the previous description data.
