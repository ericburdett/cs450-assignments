library(datasets);
library(cluster);

states = state.x77;

# Agglomerative Hierarchical Clustering

# Different options:
# - Without Scaling
# - With Scaling
# - Without Area
# - Only Frost
distance = dist(as.matrix(states));

agglomerative = hclust(distance);

png(filename="C:/Users/eric.BLAN/Desktop/h_full_unscaled.png", width=10, height=6, units='in', res=800)

plot(agglomerative, main = 'States Without Scaling', xlab = 'Distance Using Agglomerative Hierarchical Clustering', sub = '');

dev.off()


# K-Means Clustering

data = scale(states);
clusters = kmeans(data, 8);


# for loop
totalDistanceValues = list();
for (i in 1:25) {
  myCluster = kmeans(data, i);
  totalDistanceValues[i] = myCluster$tot.withinss;
}

print(totalDistanceValues);
png(filename="C:/Users/eric.BLAN/Desktop/elbow.png", width=6, height=6, units='in', res=800);
plot(1:25, totalDistanceValues, main = 'Total Sum of Squares vs K-Value', xlab = 'K-Value', ylab = 'Total Sum of Squares');
dev.off();


print("Cluster Summary:");
summary(clusters);

print("Cluster Centers:");
print(clusters$centers);

print("Cluster Assignments:");
print(clusters$cluster);





print("Cluster within sum of squares:");
print(clusters$withinss);

print("Cluster within total sum of squares:");
print(clusters$tot.withinss);

png(filename="C:/Users/eric.BLAN/Desktop/k_cluster_8.png", width=10, height=6, units='in', res=800);
clusplot(data, clusters$cluster, color=TRUE, shade=TRUE, labels=2, lines=0, main = 'States With 8 Clusters', xlab = 'K-Means Clustering', ylab = '', sub = '');
dev.off();