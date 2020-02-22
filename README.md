# Converter
this library converts grayscale images in numpy arrays for classification purposes.

# How it works
- Install the requirements
```python
pip install -r requirements
```
- Add the sorted images under the data folder
- Update the "data_folder" variable in converter.py file
- Execute the script with the following command:
```python
python converter.py
```
-Under the output folder, you will find the input data split into test and training purpose. Each folder contains two numpy arrays (data and labels)
- You can use the generated dataset with the following algorithm: https://github.com/karim007/logistic_regression

