{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## OpenStreetMap Data Case Study: Udacity’s Data Analyst Nanodegree, Project 3\n",
    "## Report"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "### Section 1: Introduction\n",
    "#### Resources Used\n",
    "- Sample projects: \n",
    "    - https://github.com/allanbreyes/udacity-data-science/tree/master/p2\n",
    "    - https://gist.github.com/carlward/54ec1c91b62a5f911c42#file-sample_project-md\n",
    "- http://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value\n",
    "- https://discussions.udacity.com/t/iterative-parsing-using-iterparse/167980/2\n",
    "\n",
    "#### Map Area: London and its Surrounding Areas, England\n",
    "- https://mapzen.com/data/metro-extracts/\n",
    "\n",
    "#### Objective\n",
    "- Audit and clean the XML Dataset in Python\n",
    "- Convert from XML to CSV\n",
    "- Load dataset into SQL and perform queries to extract information from the dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Section 1: Data Auditing and Cleaning\n",
    "Investigate the most common tags in the dataset. Most common tags were found to be:\n",
    "\n",
    "\n",
    "    [('note', 1699), ('ref', 1746), ('landuse', 1831), ('source:building', 1890), ('foot', 2092), ('operator', 2154), ('surface', 2294), ('barrier', 2627), ('lit', 2733), ('amenity', 2872), ('addr:city', 2924), ('maxspeed', 2959), ('natural', 3859), ('addr:street', 5329), ('addr:housenumber', 5621), ('created_by', 6417), ('source', 12896), ('name', 13067), ('building', 16914), ('highway', 17564)]\n",
    "\n",
    "\n",
    "For the most common tag values above, I investigated the attribute values to understand the data cleaning tasks required by creating a defaultdict to store values and frequencies of the tags highway, building, street and source."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### The Highway Tag\n",
    "Printing out the DefaultDict for the Highway Tag showed that the data was\n",
    "- clean.\n",
    "- valid --> all entries are expected for a highway\n",
    "- and consistent. There is only a single instance for each highway attribute value.\n",
    "\n",
    "Snippet of highway types:\n",
    "\n",
    "    access: 2\n",
    "    bridleway: 174\n",
    "    bus_guideway: 1\n",
    "    bus_stop: 1125\n",
    "    construction: 12\n",
    "    corridor: 1\n",
    "    crossing: 514\n",
    "    cycleway: 411\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### The Building Tag\n",
    "Printing out the DefaultDict for the Building Tag showed that the data could be made more consistent and uniform by consolidating values such as 'garage' and 'garages' and 'boat_house' and 'boathouse' as one. \n",
    "\n",
    "Cleaning Performed:\n",
    "- parsed the XML File, cleaned the building values and wrote the corrected XML document out to file.\n",
    "\n",
    "Consolidated values:\n",
    "\n",
    "    boat_house & boathouse --> boathouse\n",
    "    garage & garages --> garage"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### The Street Tag\n",
    "Some cleaning of the Street Data Required.\n",
    "- Multiple instances of the same street type need to be consolidated:\n",
    "    - Ave, ave, Avenue\n",
    "    - Rd, road, Road, ROAD\n",
    "    - St, Street, street\n",
    "    \n",
    "Cleaning Performed:\n",
    "- consolidated all instances of Avenue, Road and Street to Avenue, Road and Street using a mapping dicitionary and looping through the the values of the street tag."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### The Source Tag\n",
    "Some cleaning of the Source Data Required. There were two problems that needed to be addressed:\n",
    "- Some values had multiple sources. In this case I split the string into multiple values using regex and created separate entries.\n",
    "- Standardised the names using a mapping key. I did this in two iterative steps.\n",
    "    - First I gathered together the major information sources and standardised their names.\n",
    "    - On a second pass, for the minor information sources not included in the mapping dictionary, I reassigned the source value to other.\n",
    "    \n",
    "The result was that the number of sources was reduced down to the following list whilst still preserving the major information sources. \n",
    "\n",
    "    Bing: 5058\n",
    "    GPS: 632\n",
    "    Knowledge: 322\n",
    "    London Borough of Southwark: 1358\n",
    "    Naptan Import: 859\n",
    "    NPE: 383\n",
    "    OS: 2052\n",
    "    Other: 713\n",
    "    Photograph: 281\n",
    "    SAS: 495\n",
    "    Survey: 1807\n",
    "    Yahoo: 442\n",
    "    "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Section 2: Export XML document to CSV "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "A script (see the code workbook) was written to prepare the dataset and export it to a CSV."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Section 3: Inserting CSV into SQL"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The python files \n",
    "- create_nodes_db.py, \n",
    "- create_nodes_tags_db.py, \n",
    "- create_ways_db.py and \n",
    "- create_ways_tags_db.py \n",
    "\n",
    "were created to import the nodes, nodes_tags, ways and ways_tags csvs into SQL (these are included in the same Github repository). The ways_nodes csv was inputted in SQL. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Section 4: Data Overview\n",
    "I performed the following queries in the command line using sqlite3. I have pasted the queries and results into the notebook below for reference. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**File Size**\n",
    "\n",
    "- The original file was 2.47 GB.\n",
    "- After taking every 40th element, the final file used for the project was 60 MB."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Number of Nodes**\n",
    "\n",
    "    \"select count(id) from nodes\" --> 280985"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Number of Ways**\n",
    "\n",
    "    \"select count(id) from ways\" --> 40199"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Number of Unique Users**\n",
    "\n",
    "    \"select count(distinct uid) from nodes\" --> 3343\n",
    "\n",
    "    \"select count(distinct uid) from ways\" --> 1767"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**How does contribution from each user vary?**\n",
    "\n",
    "**Top 10 contributors**\n",
    "\n",
    "    \"create view top_10 contributors as select uid, count(id) as count from nodes group by uid order by count desc limit 10;\"\n",
    "\n",
    "**Sum up the contribution of these top 10 contributors**\n",
    "\n",
    "    \"select uid, sum(count) as sum from top_10_contributors;\"\n",
    "\n",
    "- 76,978 out of a total of 280,895 nodes are made by 10 users. \n",
    "- 50 users out of a total of 3343 who contributed to nodes created 169,204 nodes, over half of the total. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**How many members only contribute one node?**\n",
    "\n",
    "    \"create view one_contribution as select uid, count (id) as count from nodes group by uid having count = 1;\"\n",
    "\n",
    "    \"select count(uid) as sum from one_contribution;\"\n",
    "\n",
    "- 1190 users, nearly a third of the total users, have contributed one node to the OSM of London and its surrounding areas."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Average number of node contributions from each user**\n",
    "\n",
    "    \"select avg(count) from (select uid, count(id) as count from nodes group by uid) as count_all;\"\n",
    "\n",
    "- 84.05"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Create a histogram of node contributions using pandas**\n",
    "\n",
    "First write file to csv and then import it below into a dataframe.\n",
    "\n",
    "     # write file to csv\n",
    "     .mode csv\n",
    "     .output node_contribution_per_user.csv\n",
    "     select count(id) from nodes group by uid;\n",
    "     \n",
    "The histogram below shows a positively skewed distribution. Most users contribute few nodes with a small minority contributing a large number."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 293,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.text.Text at 0x2c3950b8>]"
      ]
     },
     "execution_count": 293,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\nFb4zMTOzkuRgYmZbOfLII7nzzjsLnQ0rMSXZNxd8+NKimwZbIV199S3U1q7N2/7Ly3syceK4vO2/\nIykrK2PRokXsscceza+8jalrGpxPJR1MzAqttnYtlZXVedv/4sX523dHI7X/WHqbN2+mU6fC9cqf\n7fHrLrxramrylhc/5jLrIAYNGsRNN93E4MGD6dWrF6NGjWLjxmTo9MmTJ7P33nuzyy67cPLJJ7Ni\nxYr67R577DE+/vGP06tXL8aOHfuRits777yT/fffnz59+vCFL3yB2traZvOyYMECjj32WPr06UPf\nvn25/vrrAdi4cSPjxo2jf//+DBgwgG9/+9ts2rQJgLvvvpsjjjhiq/2UlZXxz38mIwSPGTOGiy66\niOOPP57u3btz6KGH8uqrrwIwfPhwQggceOCBdO/enRkzZjSatzlz5jBw4EAmTZrErrvuyh577MG0\nadPql2/cuJHvfve7VFRU0LdvX775zW/y/vvvb7XtjTfeSN++fTn33HMbPU5z5Xn00Uc54IAD6N69\nOwMHDuSHP/xh/XoPP/wwn/zkJ+nVqxeHH344zz33XP2yQYMGceONNzJ48GB22mkntmzZ0mge2pOD\niVkHMmPGDGbNmsWrr77Ks88+y1133cXjjz/OFVdcwf3338+KFSsoLy9n5MiRAKxatYovf/nLXHfd\ndaxatYo999yTJ598sn5/Dz74INdffz0PPPAAK1eu5IgjjmDUqFFN5mH9+vUcc8wxfPGLX2TFihUs\nWrSIo48+GoBrr72WuXPnMn/+fJ599lnmzp3LtddeW79t5t1F5vwvf/lLampqWLt2LXvuuSdXXnkl\nkPzIAzz33HOsW7eO0047rck8vv7666xevZrly5dz11138bWvfY2XX34ZgEsvvZRFixYxf/58Fi1a\nxLJly5g4ceJW265du5ba2lpuv/32Jo/TVHnOP/98Jk+ezLp163j++ec56qijAJg3bx7nnXcekydP\nZvXq1Xz961/nxBNPrA+6ANOnT2fmzJmsXbuWsrLi+Bkvjly0gt8zMfuoSy65hN12242ePXtywgkn\nMG/ePO655x7OO+88Bg8eTOfOnZk0aRJPPfUUtbW1zJw5k0984hN86UtfolOnTowbN47dd9+9fn+3\n3XYbl19+Ofvssw9lZWVcdtllPPPMMyxdurTRPDz88MP07duXcePG0aVLF7p168YhhxwCwLRp05gw\nYQJ9+vShT58+TJgwgalTpza6r8y7pC996UsMGTKEsrIyzjzzTJ555pkm12+MJK655ho6d+7MZz/7\nWUaMGMF9990HJHdxN998Mz169KBbt25cdtll3HvvvfXbdurUiZqaGjp37sz222+f1fEayl+XLl1Y\nsGAB77zzDj169OCggw6qP/43vvENDj74YCQxevRott9+e5566qn6bS+55BL69euX9fE7RK/B+eJe\ng80+arfddquf7tq1K+vXr2fFihVUVHw4cna3bt3o3bs3y5YtY/ny5QwcOHCrfaTnlyxZwiWXXELv\n3r3p3bs3ffr0QRLLljU+/NDSpUvZc889G1y2fPlyyss/HMqooqKC5cuXZ12+dKCrK19r9OrVix12\n2OEj+Vi5ciXvvvsuQ4YMqS/zF77wBd566636dXfddVc6d+7cquOm/epXv+KRRx6hoqKCI488sj5Y\nLFmyhJtuuqn++L169eK1117b6jwNGDCgRcdqj16DSzaYmFnzJNGvX7+t+mnasGEDb731Fv3796dv\n374fqQNJ33UMHDiQ2267jdWrV7N69WrWrFnD+vXrGTZsWKPHHDhwIK+88kqDy/r3779V/2ZLliyh\nX79+QBLk3n333fplr7/+eovK2hJr1qzhvffeq5+vra2lX79+7LLLLnTt2pUFCxbUl3nt2rW8/fbb\n9etmW9HfUHnS2w4ZMqT+8eFJJ53E6aefDiTn78orr/zIOT/jjDNanIf25GBi1sGNGjWKu+66i/nz\n5/P+++9zxRVXMGzYMMrLyxkxYgQLFy7kgQceYPPmzdx6661b/Yh/4xvf4LrrrmPhwoUAvP3229x/\n//1NHu/444/n9ddf50c/+hEbN25k/fr1zJ07F4CRI0dy7bXXsmrVKlatWsU111zD6NGjARg8eDAL\nFiyoz2dNTU2LfjR33333+srt5tSNqrhp0yb+/Oc/88gjj3D66acjiQsuuIBx48axcuVKAJYtW8as\nWbOyzkedhspTZ9OmTUybNo1169bRqVMndt555/pWWRdccAE///nP68/Zhg0bePTRR9mwYUOL89Ce\nSrZpsFkxKC/vmdfmu+XlPbNet7Ef3qOOOoprrrmGU045hbVr1/KZz3yG6dOnA9CnTx9mzJjB2LFj\nGTNmDKNHj+bwww+v3/bkk09mw4YNjBw5ktraWnr06MExxxzDqaee2mg+dtppJx577DEuvvhiqqur\n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\ns5z9f+P8E7tgLju3AAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x126b9a58>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "% matplotlib inline\n",
    "\n",
    "df = pd.read_csv('node_contribution_per_user.csv', header=None, names = ['node_count_per_user'])\n",
    "df\n",
    "ax = df.plot(kind = 'hist', title = 'Number of Nodes created per User',logy=True,alpha = 0.5, bins = 20)\n",
    "ax.set(xlabel = 'Number of Nodes Created')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The histogram above showed us the contribution distribution by user. Below I have looked at the contribution distribution over time, to understand when most of the contributions to the London OSM Dataset were made.\n",
    "\n",
    "It seems like contributions for the number of nodes in London grew steadily up to 2012 but have since declined."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 317,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.text.Text at 0x2187c6d8>, <matplotlib.text.Text at 0x2de8bef0>]"
      ]
     },
     "execution_count": 317,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<matplotlib.figure.Figure at 0x20e594a8>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import csv\n",
    "import re\n",
    "\n",
    "pattern = re.compile(r'(\\w+)')\n",
    "\n",
    "data = []\n",
    "with open ('date.csv', 'rb') as csvfile:\n",
    "    reader = csv.reader(csvfile)\n",
    "    for row in reader:\n",
    "        row = str(row)\n",
    "        result = (pattern.search(row)).group()\n",
    "        data.append(result)\n",
    "        \n",
    "df = pd.DataFrame(data)\n",
    "ax = df[0].value_counts().sort_index(axis=0, ascending=True).plot(kind = 'bar', title = 'Node Creation per Years')\n",
    "ax.set(xlabel = 'Year', ylabel = 'Nodes Created')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**How many nodes on average make up a way?**\n",
    "\n",
    "    query = \"select avg(count) from (select ways_tags.id, count(node_id) as count from \n",
    "             ways_tags join ways_nodes on ways_tags.id = ways_nodes.id \n",
    "             group by ways_tags.id) as subq\"\n",
    "\n",
    "**Result:**\n",
    "23.1\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Let's now use the data to understand more about London."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**What are the ten most common amenities?**\n",
    "\n",
    "    query = \"select value, count(value) as count from nodes_tags \n",
    "            where key = 'amenity' group by value order by count desc limit 10;\"\n",
    "\n",
    "**Result:**\n",
    "\n",
    "- post_box,334\n",
    "- bench,196\n",
    "- bicycle_parking,151\n",
    "- pub,138\n",
    "- restaurant,113\n",
    "- telephone,97\n",
    "- cafe,75\n",
    "- waste_basket,70\n",
    "- place_of_worship,65\n",
    "- fast_food,62\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Additional Ideas"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- We saw above that few users contribute a large proportion of the nodes created in the dataset. A SQL query showed that 10 of the top contributors created nearly a quarter of the total nodes. To try to increase the number of contributors, we could look at ways of making it easier to contribute. Perhaps the process could be incorporated into a game such as Pokemon Go, where users who are already roaming the streets are awarded points for identifying and labelling features on the map. \n",
    "- I was impressed by Google's use of computer vision to extract information from their streetview photos and embed them in their maps. Perhaps, users could also share their photographs and algorithms could be developed to extract information from these. \n",
    "- Regarding the dataset, I would be interested in seeing whether way creation activity had increased after 2012 when we saw the beginning of the decline in node creation. Perhaps the contributors changed their focus from creating the map to adding features to it. \n",
    "\n",
    "\n",
    "\n",
    "\n",
    "- Data cleaning aims to address imperfections in data quality, of which there are five main features\n",
    "    - Validity\n",
    "    - Accuracy\n",
    "    - Completeness\n",
    "    - Consistency\n",
    "    - Uniformity\n",
    "\n",
    "- Much of the effort above has been directed towards validity - deciding what street name, or building type name is valid and checking that the field values adhere to these constraints. Other work could look at auditing accuracy of data by comparing our dataset to another. We could extract google maps information from the API, and use it as a gold standard dataset to check for accuracy of the Open Street Map Dataset. This would also allow us to check for completeness. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Conclusion"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- The Open Street Map Data for London was downloaded in XML format, sampled for every 40th element. \n",
    "- Some parts of the data were audited and cleaned although cleaning tasks remain. The data was written to a CSV file, loaded into a SQL database (sqlite) and queries were constructed to understand the dataset (the number of nodes, ways, the contribution distributions by user and time) as well as extract information about London. "
   ]
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python [Root]",
   "language": "python",
   "name": "Python [Root]"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}