import data
import clustering

# Documentation and links we used
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.columns.html
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mean.html
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.apply.html
# https://www.geeksforgeeks.org/pandas-series-dt-year/
# https://docs.python.org/3/library/datetime.html
# https://www.geeksforgeeks.org/python-pandas-apply/

print("Part A: ")
df = data.load_data('london.csv')

df = data.add_new_columns(df)

df =  data.data_analysis(df)

# Second exercise
print()
print("Part B: ")
df = data.load_data('london.csv')
features = ['cnt', 't1']
data = clustering.transform_data(df, features)
for k in {2, 3, 5}:
    labels, centroids = clustering.kmeans(data, k)
    print(f"k = {k}")
    clustering.visualize_results(data, labels, centroids, 'plots.png')
    print()
