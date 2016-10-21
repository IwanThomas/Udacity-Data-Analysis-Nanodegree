## Summary

This data visualisation plots the survival rate of passengers onboard the Titanic sorted by gender and passenger class. The data can be downloaded from [here] (https://www.udacity.com/api/nodes/5420148578/supplemental_media/titanic-datacsv/download).

## Design

I downloaded the data and used techniques from the Exploratory Data Analysis (EDA) course to explore the dataset. This investigation showed that gender and passenger class were key factors that influenced the survival rate of a given passenger. I wanted to plot the survival rate of male and female passengers in classes 1, 2 and 3. A bar chart was the most suitable choice for this, enabling easy comparison between the different categorical variables.

I created an initial design which can be seen below. The accompanying html code can be found in index1.html.

![Alt text](https://github.com/IwanThomas/Udacity-Data-Analysis-Nanodegree/blob/master/Project-6/Images/index1_screenshot.PNG)

## Feedback 

I gathered feedback from three friends. Here is a summary of what they told me:
- The chart needs a title.
- It would be better to see the survival rate instead of just the raw count.
- The chart order and legend order should match.

## Post-feedback Design

Following on from the feedback, I implemented the following changes
- I added a title to my chart
- I changed the y axis from survival count to survival rate
- I matched the ordering of the charts with the ordering of the legend.

Below is the final rendition of my data visualisation. The html code can be found in index2.html.

![Alt text](https://github.com/IwanThomas/Udacity-Data-Analysis-Nanodegree/blob/master/Project-6/Images/index2_screenshot.PNG)

## Resources

- dimple.js Documentation
- Stack Overflow
- Udacity Course

