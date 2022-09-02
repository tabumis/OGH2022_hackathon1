library(cluster)
library(dplyr)
library(ggplot2)
library(readr)
library(Rtsne)
library(dplyr)
library(e1071)


train_data <- read_csv("C:/OpenGeoHub2022/lcv_pasture_classif.matrix.train_2000..2020_brazil.eumap_summer.school.2022.csv",
                       col_types = cols(...1 = col_skip(), date = col_date(format = "%Y-%m-%d")))

val_data <- read_csv("C:/OpenGeoHub2022/lcv_pasture_classif.matrix.val_2000..2020_brazil.eumap_summer.school.2022.csv",
                     col_types = cols(...1 = col_skip(), date = col_date(format = "%Y-%m-%d")))


test_data <- read_csv("C:/OpenGeoHub2022/lcv_pasture_classif.matrix.test_2000..2020_brazil.eumap_summer.school.2022.csv",
                      col_types = cols(...1 = col_skip(), date = col_date(format = "%Y-%m-%d")))


table(train_data$area)
table(val_data$area)


train_data %>%
  subset(., area=="brazil") %>%
  do(h = hist(.$class))

train_data %>%
  subset(., area=="eumap") %>%
  do(h = hist(.$class))


val_data %>%
  subset(., area=="brazil") %>%
  do(h = hist(.$class))

val_data %>%
  subset(., area=="eumap") %>%
  do(h = hist(.$class))


# As it is seen the datasets are unbalanced in terms of obsevations per each class.
# In addition the train dataset is unbalanced in representing 'area'
# My suggested solution is to compile a new train and validation samples, and dissaggregate class 3 to smaller clusters of sub-classes using k-means applied on temporal spectral bands (landsat)


# Creating new train and val samples with more balanced representations of area and classes
df<-rbind(train_data,val_data)


df1 <- df %>%
  dplyr:: select(contains(c("tile","area","year","latitude", "longitude","class","fire","landsat")))

tr_class1<-filter(df1, class== 1) %>% sample_n(., 400) #randomly choosing n of observtions per class
tr_class2<-filter(df1, class== 2) %>% sample_n(., 250)
tr_class3<-filter(df1, class== 3) %>% sample_n(., 1200)

traindf<-do.call("rbind", list(tr_class1, tr_class2, tr_class3))

traindf$year<-as.factor(traindf$year)

#-----------------------------------------------------------------
# diassaggregating class 3 (per each area) to synthetic subclasses using landstar spectral bands and year
brazil_class3<-subset(traindf, area=='brazil' & class==3)
eumap_class3<-subset(traindf, area=='eumap' & class==3)


gower_dist <- daisy(brazil_class3[,-c(1:3,5:7,9:11)], metric = "gower")
gower_mat <- as.matrix(gower_dist)

sil_width <- c(NA)
for(i in 2:8){  # identifying optimal n of clusters using siluethe method
  pam_fit <- pam(gower_dist, diss = TRUE, k = i)
  sil_width[i] <- pam_fit$silinfo$avg.width
}
plot(1:8, sil_width,
     xlab = "Number of clusters",
     ylab = "Silhouette Width")
lines(1:8, sil_width)


br_clusters<-kmeans(brazil_class3[,-c(1:3,5:7,9:11)], 5, iter.max = 10, nstart = 1) # getting 5 clusterss for class 3 for Brazil domain


brazil_class3$clusters<-br_clusters[["cluster"]]

brazil_class3$subclass<-0

brazil_class3$subclass<-ifelse(brazil_class3$clusters==1, 31,brazil_class3$subclass) # reclassification
brazil_class3$subclass<-ifelse(brazil_class3$clusters==2, 32,brazil_class3$subclass)
brazil_class3$subclass<-ifelse(brazil_class3$clusters==3, 33,brazil_class3$subclass)
brazil_class3$subclass<-ifelse(brazil_class3$clusters==4, 34,brazil_class3$subclass)
brazil_class3$subclass<-ifelse(brazil_class3$clusters==5, 35,brazil_class3$subclass)

hist(brazil_class3$subclass)

brazil_class3$class<-ifelse(brazil_class3$class==3,brazil_class3$subclass,brazil_class3$class)


# replicating the above steps for Europe domain
gower_dist <- daisy(eumap_class3[,-c(1:3,5:7,9:11)], metric = "gower")
gower_mat <- as.matrix(gower_dist)


eu_clusters<-kmeans(eumap_class3[,-c(1:2,4:10)], 5, iter.max = 10, nstart = 1)


eumap_class3$clusters<-eu_clusters[["cluster"]]

eumap_class3$subclass<-0

eumap_class3$subclass<-ifelse(eumap_class3$clusters==1, 41,eumap_class3$subclass)
eumap_class3$subclass<-ifelse(eumap_class3$clusters==2, 42,eumap_class3$subclass)
eumap_class3$subclass<-ifelse(eumap_class3$clusters==3, 43,eumap_class3$subclass)
eumap_class3$subclass<-ifelse(eumap_class3$clusters==4, 44,eumap_class3$subclass)
eumap_class3$subclass<-ifelse(eumap_class3$clusters==5, 45,eumap_class3$subclass)


hist(eumap_class3$subclass)

eumap_class3$class<-ifelse(eumap_class3$class==3,eumap_class3$subclass,eumap_class3$class)

# binding a new train and val datasets
new_df<-rbind(eumap_class3,brazil_class3)

new_df<-new_df[,-c(96,97)]

traindf<-do.call("rbind", list(tr_class1, tr_class2, new_df)) # new train dataset
valdf<-subset(df1, !(latitude %in% traindf$latitude & longitude %in% traindf$longitude & year %in% traindf$year)) # building a validation dataset



#---------------------------------------------------------
#building a classification model

aa<-traindf[,-c(1,6:7, 9:10)] # excluding redundant features from training dataset
aa$area<-as.factor(aa$area)
aa$year<-as.factor(aa$year)
aa$class<-as.factor(aa$class)

aa<-na.omit(aa)



valdf$area<-as.factor(valdf$area)
valdf$year<-as.factor(valdf$year)
valdf$class<-as.factor(valdf$class)

# Determining optimal cost and gamma parameters for SVM (RBF)

tuned_parameters<- tune(svm, class~., data = aa, kernel="radial",
                        ranges = list(gamma=c(0.015,0.018,0.02,0.022),
                                      cost = c(3.5,3.6,3.75,3.85)
                        ))

tuned_parameters[["best.parameters"]]



library(e1071)
svm_model <- svm(formula = class ~., data = aa, type = "C-classification",kernel= "radial",
                 gamma=tuned_parameters[["best.parameters"]][["gamma"]],cost=tuned_parameters[["best.parameters"]][["cost"]],
                 scale = T)
summary(svm_model)

predicted<- predict(svm_model, valdf) # test the model on validation dataset
predicted<-predicted %>%
  as.integer(.) %>%
  ifelse(.>2,3,.) %>% # agrregating all sub-classes to class 3
  as.factor(.)


library(caret)
caret::confusionMatrix(predicted,valdf$class)
MLmetrics::F1_Score(predicted,valdf$class)


#-------------------------------------------------
# applying the tuned model on a test dataset

testdf<-test_data %>%
  dplyr:: select(contains(c("tile","area","year","latitude", "longitude","class","fire","landsat")))

testdf<-testdf[,-c(1,6:7)]
testdf$area<-as.factor(testdf$area)
testdf$year<-as.factor(testdf$year)

test_predicted<- predict(svm_model, testdf) # test the model on validation dataset
test_predicted<-test_predicted %>%
  as.integer(.) %>%
  ifelse(.>2,3,.)

test_data$class<- test_predicted

hist(test_data$class)

write.csv(test_data,"C:/OpenGeoHub2022/hackathon1_results_au.csv")
