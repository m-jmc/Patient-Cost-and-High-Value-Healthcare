# DRG Data: https://data.cms.gov/inpatient-provider-lookup/view-data
# HSP Data: https://data.medicare.gov/Hospital-Compare/Medicare-Hospital-Spending-by-Claim/nrth-mfg3
# Provider (SER): https://data.medicare.gov/Hospital-Compare/Payment-and-value-of-care-Hospital/c7us-v4mf
# Zip: https://public.opendatasoft.com/explore/dataset/us-zip-code-latitude-and-longitude/export/
# DRG to MDC:


#########################################
#                                       #
# NOTE
# Before loading data sets, remove all
# spaces from the header columns of the
# .csv or you're gonna have a bad time.
# 
# Name the data frames as shown 
# in read.csv command
#                                       #
#########################################

library(caret)
library(tidyverse)
library(DataExplorer)
library(xgboost)
library(ggcorrplot)
library(readr)
library(reshape2)
library(fastDummies)
library(WVPlots)
library(ROCR)
library(gridExtra)
library(reshape2)


# hspdata <- read_csv("Data/Data/Medicare_Hospital_Spending_by_Claim.csv")
# drgdata <- read_csv("Data/Data/Medicare_Provider_Charge_Inpatient_DRGALL_FY2016 - no space.csv")
# serdata <- read_csv("Data/Data/Provider Value of Care - Hospital - nospace.csv")
# zipdata <- read_csv("Data/zipdata.csv")
# DRGtoMDC <- read_csv("C:/Users/hfmm4140/Downloads/Hackathon/Data/Data/DRGtoMDCfulltable.csv")

hspdata <- read_csv("Medicare Spending by Claim.csv")
drgdata <- read_csv("Medicare_Provider_Charge_Inpatient_DRGALL_FY2016.csv")
serdata <- read_csv("Provider Value of Care - Hospital.csv")
zipdata <- read_csv("Zip.csv")
DRGtoMDC <- read_csv("DRGtoMDCfulltable.csv")



#Create column of FH locations in dataset
drgdata$FH <- (ifelse(drgdata$ProviderId == '520177', "Yes", 
                      (ifelse(drgdata$ProviderId == '520103', "Yes", 
                              (ifelse(drgdata$ProviderId == '520063', "Yes", "No"))))))

#parse the drg ID into the first 3 numbers as drg ID
drgdata$drgID <- substr(drgdata$DRGDefinition, start = 1, stop = 3)

##To create column with FH DRG's' in the drgdata dataset
#drgdata$FHdrgID <- (ifelse(drgdata$FH %in% "Yes", drgdata$drgID, ""))

#subset drgs in the FH data select=c(drgID,ProviderName) to add hospital name
FHdrgs <- subset(drgdata, FH %in% "Yes", select=c(drgID)) 

#create flag of all drgs that match with FH data in the dataset (******************************This might not work as expected?)
#might want to try the following: drgdata$MDC <- DRGtoMDc$MDC[match(drgdata$drgID, DRGtoMDc$`MS-DRG`)]
drgdata$FHdrgflag <- (ifelse(drgdata$drgID %in% FHdrgs$drgID, "Yes", "No"))

#create flag of all drgs matching FH data from wisconsin
drgdata$WI <- (ifelse(drgdata$ProviderState == 'WI' & drgdata$FHdrgflag == 'Yes', "Yes", "No"))

#Convert dollars to numeric values: 
drgdata$ACC <- parse_number(drgdata$AverageCoveredCharges)
drgdata$ATP <- parse_number(drgdata$AverageTotalPayments)
drgdata$AMP <- parse_number(drgdata$AverageMedicarePayments)

#Find average per discharge (ex: ACC/Total discharge)

drgdata$ACCperDischarge <- (drgdata$ACC / drgdata$TotalDischarges)
drgdata$ATPperDischarge <- (drgdata$ATP / drgdata$TotalDischarges)
drgdata$AMPperDischarge <- (drgdata$AMP / drgdata$TotalDischarges)


#Find % of total bill payed by CMS AMP/ACC * 100, rounded two places

drgdata$PercentCMSperDischarge <- ((drgdata$AMPperDischarge / drgdata$ACCperDischarge) *100)
drgdata$PercentCMSperDischarge <- format(round(drgdata$PercentCMSperDischarge, 2), nsmall = 2)


#########################################
#                                       #
# Provider Quality Data Transformation  #
#                                       #
#########################################


#Create numerical scores based off value of care category
#See Quality Rank score Table reference below for detail
serdata$QualityScore <- (ifelse(grepl("Not Available", serdata$Valueofcarecategory), 'N/A',
                                ifelse(grepl("Average mortality and average payment", serdata$Valueofcarecategory), '5',
                                       ifelse(grepl("Average mortality and higher payment", serdata$Valueofcarecategory), '2.5',
                                              ifelse(grepl("Average complications and average payment", serdata$Valueofcarecategory), '5',
                                                     ifelse(grepl("Average mortality and lower payment", serdata$Valueofcarecategory), '7.5',
                                                            ifelse(grepl("Worse complications and higher payment", serdata$Valueofcarecategory), '0',
                                                                   ifelse(grepl("Worse mortality and average payment", serdata$Valueofcarecategory), '2.5',
                                                                          ifelse(grepl("Better mortality and higher payment", serdata$Valueofcarecategory), '5',
                                                                                 ifelse(grepl("Average complications and higher payment", serdata$Valueofcarecategory), '2.5',
                                                                                        ifelse(grepl("Average complications and lower payment", serdata$Valueofcarecategory), '7.5',
                                                                                               ifelse(grepl("Worse mortality and higher payment", serdata$Valueofcarecategory), '0',
                                                                                                      ifelse(grepl("Better mortality and average payment", serdata$Valueofcarecategory), '7.5',
                                                                                                             ifelse(grepl("Worse mortality and lower payment", serdata$Valueofcarecategory), '5',
                                                                                                                    ifelse(grepl("Better mortality and lower payment", serdata$Valueofcarecategory), '10',
                                                                                                                           ifelse(grepl("Better complications and lower payment", serdata$Valueofcarecategory), '10',
                                                                                                                                  ifelse(grepl("Worse complications and lower payment", serdata$Valueofcarecategory), '5',
                                                                                                                                         ifelse(grepl("Better complications and average payment", serdata$Valueofcarecategory), '7.5',
                                                                                                                                                ifelse(grepl("Worse complications and average payment", serdata$Valueofcarecategory), '2.5',
                                                                                                                                                       ifelse(grepl("Better complications and higher payment", serdata$Valueofcarecategory), '5', 'NULL'))))))))))))))))))))


serdata$QualityScore <- as.numeric(serdata$QualityScore)

#Designate Payment measure ID with MDc value (MDc ID xxx DRG Range)
#04	Diseases and Disorders of the Respiratory System	163 - 208
#05	Diseases and Disorders of the Circulatory System	215 - 316
#08	Diseases and Disorders of the Musculoskeletal System And Connective Tissue	453 - 566

serdata$MDC <- (ifelse(grepl("PAYM_30_AMI", serdata$PaymentmeasureID), '05',
                       ifelse(grepl("PAYM_30_HF", serdata$PaymentmeasureID), '05',
                              ifelse(grepl("PAYM_90_HIP_KNEE", serdata$PaymentmeasureID), '08',
                                     ifelse(grepl("PAYM_30_PN", serdata$PaymentmeasureID), '04', 'N/A')))))
#measure name might be handy for graphs
serdata$measurename <- (ifelse(grepl("PAYM_30_AMI", serdata$PaymentmeasureID), 'Heart Attack',
                               ifelse(grepl("PAYM_30_HF", serdata$PaymentmeasureID), 'Heart Failure',
                                      ifelse(grepl("PAYM_90_HIP_KNEE", serdata$PaymentmeasureID), 'Hip and Knee',
                                             ifelse(grepl("PAYM_30_PN", serdata$PaymentmeasureID), 'Pneumonia', 'N/A')))))


#Map DRGs to their MDc
drgdata$MDC <- DRGtoMDC$MDC[match(drgdata$drgID, DRGtoMDC$`MS-DRG`)]

#create subset of needed columns to merge with drg dataset
colnames(serdata)[colnames(serdata)=="ProviderID"] <- "Provider_ID"
serdata.subset <- serdata[c("Provider_ID","QualityScore","MDC","measurename")]

#Left Join DRG and SER subset columns
colnames(drgdata)[colnames(drgdata)=="ProviderId"] <- "Provider_ID"
drgdata$Provider_ID <- as.numeric(drgdata$Provider_ID)
drgdata$PercentCMSperDischarge <- as.numeric(drgdata$PercentCMSperDischarge )
drg.ser <- drgdata %>% left_join(serdata.subset, by=c("Provider_ID", "MDC"))

###############################
#Additional Analysis if needed#
###############################
FH.drg.ser <- drg.ser %>% filter(FH == "Yes")
WI.drg.ser <- drg.ser %>% filter(ProviderState == "WI")


fh.drg.ser.plot <- ggplot(data=subset(FH.drg.ser, !is.na(QualityScore)), aes(x=QualityScore, y=PercentCMSperDischarge, color=measurename, fill=measurename)) + geom_jitter() + scale_y_continuous() +
  scale_color_manual(values=c("#ff0000", "#ffaa00","#696870","#0d00ff")) +
  labs(title="Quality score by percent CMS rembursement and measure (FH)")

fh.drg.ser.plot

wi.drg.ser.plot <- ggplot(data=subset(WI.drg.ser, !is.na(QualityScore)), aes(x=QualityScore, y=PercentCMSperDischarge, color=measurename, fill=measurename)) + geom_jitter() + scale_y_continuous() +
  scale_color_manual(values=c("#ff0000", "#ffaa00","#696870","#0d00ff")) +
  labs(title="Quality score by percent CMS rembursement and measure (WI)")

wi.drg.ser.plot


#########################################
#                                       #
# Hospital Data Transformation          #
#                                       #
#########################################


#Can I break down the differences in percent of spending with the value from the provider dataset
#cant really relate it to the DRG, but if we find where we have the most opportunity it could differentiate the high and low value
#performers according to % of spend?

colnames(hspdata)[colnames(hspdata)=="PROVIDER_ID"] <- "Provider_ID"

#Shorten column values for recasting
hspdata$PERIOD <- (ifelse(grepl("1 to 3 days Prior to Index Hospital Admission", hspdata$PERIOD), 'Before',
                          ifelse(grepl("During Index Hospital Admission", hspdata$PERIOD), 'During',
                                 ifelse(grepl("1 through 30 days After Discharge from Index Hospital Admission", hspdata$PERIOD), 'After',
                                        ifelse(grepl("Complete Episode", hspdata$PERIOD), 'Total', 'N/A')))))


hspdata$CLAIM_TYPE <- (ifelse(grepl("Home Health Agency", hspdata$CLAIM_TYPE), 'HomeHealth',
                              ifelse(grepl("Skilled Nursing Facility", hspdata$CLAIM_TYPE), 'SkilledRN',
                                     ifelse(grepl("Durable Medical Equipment", hspdata$CLAIM_TYPE), 'DME',
                                            ifelse(grepl("Inpatient", hspdata$CLAIM_TYPE), 'IP',
                                                   ifelse(grepl("Outpatient", hspdata$CLAIM_TYPE), 'OP',
                                                          ifelse(grepl("Carrier", hspdata$CLAIM_TYPE), 'Carrier', 
                                                                 ifelse(grepl("Hospice", hspdata$CLAIM_TYPE), 'Hospice',
                                                                        ifelse(grepl("Total", hspdata$CLAIM_TYPE), 'Total', 'N/A')))))))))


#recast long form into wide
hspdata.original <- hspdata
hsp.recast <- recast(hspdata, Provider_ID ~ variable + CLAIM_TYPE + PERIOD, id.var = c("Provider_ID", "CLAIM_TYPE", "PERIOD"))
hsp.no.duplicate <- hsp.recast[!duplicated(as.list(hsp.recast))]



#########################################
#                                       #
#        Getting a Little Boost         #
#                                       #
#########################################

#Providervalues <- drg.ser[c("Provider_ID","HighValue")]
#HSP.xgb <- hsp.no.duplicate[-c(2,3,108,109)]

#despite converting the long form to wide above, I'm going to redo it here expressly for model creation to make removing columns easier

HSP.xgb <- subset(hspdata.original, select = -c(HOSPITAL_NAME, STATE, START_DATE, END_DATE,
                                                AVG_SPNDG_PER_EP_STATE,AVG_SPNDG_PER_EP_NATIONAL, PERCENT_OF_SPNDG_STATE,
                                                PERCENT_OF_SPNDG_NATIONAL))

hsp.recast.xgb <- recast(HSP.xgb, Provider_ID ~ variable + CLAIM_TYPE + PERIOD, id.var = c("Provider_ID", "CLAIM_TYPE", "PERIOD"))

#comparison <- setdiff(hsp.recast.xgb, HSP.xgb)
#Error: not compatible: Cols in x but not y: `PERCENT_OF_SPNDG_HOSPITAL_SkilledRN_During`, `AVG_SPNDG_PER_EP_HOSPITAL_OP_During`, 
#`AVG_SPNDG_PER_EP_HOSPITAL_Hospice_During`, `AVG_SPNDG_PER_EP_HOSPITAL_SkilledRN_During`, `PERCENT_OF_SPNDG_HOSPITAL_Hospice_During`, 
#`PERCENT_OF_SPNDG_HOSPITAL_OP_During`
#Those columns all contain duplicate data but shouldn't be removed, so just rename the data frame. 
#HSP.xgb <- hsp.recast.xgb[!duplicated(as.list(hsp.recast.xgb))]
HSP.xgb <- hsp.recast.xgb

#Average value scores by provider (funs is depreicated, but list is generating an error, I'm going to just live with this for now)
serdata.sub.xgb <- serdata.subset %>% group_by(Provider_ID) %>% summarise_at(vars(-MDC,-measurename), funs(mean(., na.rm=TRUE)))

serdata.sub.xgb$QualityScore <- as.numeric(serdata.sub.xgb$QualityScore)
#high Value care boolean for values >7:
serdata.sub.xgb$Highvalue <- (ifelse(serdata.sub.xgb$QualityScore >= 7, 1,0))

#Join the high value flag with the HSP data and remove the quality score column (no longer needed, easier to do it here)
HSP.xgb <- HSP.xgb %>% left_join(serdata.sub.xgb, by=c("Provider_ID"))
#HSP.xgb <- subset(HSP.xgb, select = -c(QualityScore))#, Provider_ID))


#remove % which creates list of factors
HSP.xgb <- lapply(HSP.xgb, gsub, pattern='%', replacement='')

#convert factors to numeric
HSP.xgb <- lapply(HSP.xgb, function(x) as.numeric(as.character(x)))

# convert list back to data frame, remove NA values
# Not a huge fan, removes 107 rows, but end up not using it in this case)

HSP.xgb <- as.data.frame(HSP.xgb)
HSP.xgb <- na.omit(HSP.xgb)
HSP.xgb$Highvalue <- as.factor(HSP.xgb$Highvalue)
HSP.xgb.w.ser <- HSP.xgb
HSP.xgb <- subset(HSP.xgb, select = -c(Provider_ID, QualityScore))

#Break into 70/30 test and training subsets:
smp_size <- floor(0.70 * nrow(HSP.xgb))
train_ind <- sample(seq_len(nrow(HSP.xgb)), size = smp_size)
train <- HSP.xgb[train_ind,]
test <- HSP.xgb[-train_ind,]

## XGBoost
y <- subset(train, select = c(Highvalue))
x <- subset(train, select = -c(Highvalue))
x <- as.matrix(x)
y <- as.matrix(y)
y <- as.numeric(y)

#                              ----------------------------ADDITIONAL XGB TUNING HERE PLEASE
# https://www.kaggle.com/c/grupo-bimbo-inventory-demand/discussion/23170

bst <- xgboost(data = x, label = y, max.depth = 6, eta = 0.3, nthread = 4, nrounds = 50,objective = "binary:logistic")

importance <- xgb.importance(feature_names = colnames(x), model = bst)
importance
xgb.plot.importance(importance_matrix = importance)

#prediction on the test set: 
pred <- predict(bst, data.matrix(test[,-45]))
#convert regression into binary classification:
prediction <- as.numeric(pred > 0.5)
err <- mean(as.numeric(pred > 0.5) != test$Highvalue)
print(paste("test-error=", err))

#confusion Matrix

confusionMatrix(as.factor(prediction), as.factor(test$Highvalue))

#ROC
pred1 <- prediction(prediction, test$Highvalue)
perf1 <- performance(pred1, "tpr", "fpr")
plot(perf1)




#########################################
#                                       #
#    This seems like a regression       #
#                                       #
#########################################

glm.data <- HSP.xgb


#check and remove zero or near zero variance features 
nzv <- nearZeroVar(glm.data, saveMetrics = FALSE)
glm.data.nzv <- glm.data[,-nzv]
glm.data.nzv <- lapply(glm.data.nzv, function(x) as.numeric(as.character(x)))
glm.data.nzv <- as.data.frame(glm.data.nzv)

# Create correlation table to find highly correlated features. Remove those with > 0.75 correlation
# to prevent multicollinearity 
glm.data.nzv.cor <- cor(glm.data.nzv)
summary(glm.data.nzv.cor[upper.tri(glm.data.nzv.cor)])
glmdnc <- findCorrelation(glm.data.nzv.cor, cutoff = .75)
glm.data.nzv.nocor <- glm.data.nzv[,-glmdnc]

# Center and Scale values
glm.data.nzv.nocor.hv <- subset(glm.data.nzv.nocor, select = c(Highvalue))
glm.data.nzv.nocor <- subset(glm.data.nzv.nocor, select = -c(Highvalue))

glm.data.preprocess <- preProcess(glm.data.nzv.nocor, method = c("center","scale"))
glm.preprocess <- predict(glm.data.preprocess,glm.data.nzv.nocor)
glm.preprocess$Highvalue <- glm.data.nzv.nocor.hv$Highvalue
glm.preprocess$Highvalue <- as.factor(glm.preprocess$Highvalue)

# Create random 60/40 training and testing dataset
smp_size <- floor(0.60 * nrow(glm.preprocess))
train_ind <- sample(seq_len(nrow(glm.preprocess)), size = smp_size)
glm.train <- glm.preprocess[train_ind,]
glm.test <- glm.preprocess[-train_ind,]


# Below, testing several different models to find the best accuracy

#glm <- glm(Highvalue~., data = glm.train, family = "binomial")
#summary(glm)


control <- trainControl(method ="cv", number = 10)
metric <- "Accuracy"
#model <- train(Highvalue~., data=glm.train, method="lvq", preProcess="scale", trControl=control)
#importance.lvq <- varImp(model, scale=FALSE)
#test <- as.data.frame(test)
#print(importance.lvq)

fit.glm <- train(Highvalue~., data = glm.train, method="glm", metric=metric, trControl=control)
fit.lda <- train(Highvalue~., data = glm.train, method="lda", metric=metric, trControl=control)
fit.cart <- train(Highvalue~., data = glm.train, method="rpart", metric=metric, trControl=control)
fit.knn <- train(Highvalue~., data = glm.train, method="knn", metric=metric, trControl=control)
fit.svm <- train(Highvalue~., data = glm.train, method="svmRadial", metric=metric, trControl=control)
fit.rf <- train(Highvalue~., data = glm.train, method="rf", metric=metric, trControl=control)
#fit.gmb <- train(Highvalue~., data = glm.train, method="gmb", metric=metric, trControl=control)
results <- resamples(list(lda=fit.lda,cart=fit.cart,knn=fit.knn, svm=fit.svm,rf=fit.rf,glm=fit.glm))
summary(results)

# dot plot comparison of accuracy from all models
scales <- list(x=list(relation="free"), y=list(relation="free"))
dotplot(results, scales=scales)

importance.rf <- varImp(fit.rf, scale = FALSE)
plot(importance.rf)
#View(fit.rf)


#Running with LDA on this one

#                   -------------------------------------REBUILD TO USE XGB FEATURES

# importance.lda <- varImp(fit.lda, scale = FALSE)
# plot(importance.lda)
# 
# # Testing the prediction on the testing data and creating a confusion matrix
# lda.prediction <- predict(fit.lda, glm.test)
# confusionMatrix(lda.prediction,glm.test$Highvalue)
# 
# # Checking performance of just the top 5 features: 
# lda.top5 <- (train(Highvalue~
#                      AVG_SPNDG_PER_EP_HOSPITAL_SkilledRN_After +
#                      PERCENT_OF_SPNDG_HOSPITAL_IP_During +
#                      PERCENT_OF_SPNDG_HOSPITAL_IP_After +
#                      PERCENT_OF_SPNDG_HOSPITAL_SkilledRN_Before +
#                      PERCENT_OF_SPNDG_HOSPITAL_Carrier_After +
#                      AVG_SPNDG_PER_EP_HOSPITAL_Hospice_After,
#                    data = glm.train, method="lda", metric = metric, trControl=control)) 
# 
# lda.top5
# 
# lda.top.prediction <- predict(lda.top5, glm.test)
# confusionMatrix(lda.top.prediction, glm.test$Highvalue)




#########################################
#                                       #
#       Split violin plotting           #
#                                       #
#########################################

GeomSplitViolin <- ggproto("GeomSplitViolin", GeomViolin, 
                           draw_group = function(self, data, ..., draw_quantiles = NULL) {
                             data <- transform(data, xminv = x - violinwidth * (x - xmin), xmaxv = x + violinwidth * (xmax - x))
                             grp <- data[1, "group"]
                             newdata <- plyr::arrange(transform(data, x = if (grp %% 2 == 1) xminv else xmaxv), if (grp %% 2 == 1) y else -y)
                             newdata <- rbind(newdata[1, ], newdata, newdata[nrow(newdata), ], newdata[1, ])
                             newdata[c(1, nrow(newdata) - 1, nrow(newdata)), "x"] <- round(newdata[1, "x"])
                             
                             if (length(draw_quantiles) > 0 & !scales::zero_range(range(data$y))) {
                               stopifnot(all(draw_quantiles >= 0), all(draw_quantiles <=
                                                                         1))
                               quantiles <- ggplot2:::create_quantile_segment_frame(data, draw_quantiles)
                               aesthetics <- data[rep(1, nrow(quantiles)), setdiff(names(data), c("x", "y")), drop = FALSE]
                               aesthetics$alpha <- rep(1, nrow(quantiles))
                               both <- cbind(quantiles, aesthetics)
                               quantile_grob <- GeomPath$draw_panel(both, ...)
                               ggplot2:::ggname("geom_split_violin", grid::grobTree(GeomPolygon$draw_panel(newdata, ...), quantile_grob))
                             }
                             else {
                               ggplot2:::ggname("geom_split_violin", GeomPolygon$draw_panel(newdata, ...))
                             }
                           })

geom_split_violin <- function(mapping = NULL, data = NULL, stat = "ydensity", position = "identity", ..., 
                              draw_quantiles = NULL, trim = TRUE, scale = "area", na.rm = FALSE, 
                              show.legend = NA, inherit.aes = TRUE) {
  layer(data = data, mapping = mapping, stat = stat, geom = GeomSplitViolin, 
        position = position, show.legend = show.legend, inherit.aes = inherit.aes, 
        params = list(trim = trim, scale = scale, draw_quantiles = draw_quantiles, na.rm = na.rm, ...))
}


#Create column of FH locations in dataset
HSP.xgb.w.ser$FH <- (ifelse(HSP.xgb.w.ser$Provider_ID == '520177', "Froedtert", 
                            (ifelse(HSP.xgb.w.ser$Provider_ID == '520103', "Froedtert", 
                                    (ifelse(HSP.xgb.w.ser$Provider_ID == '520063', "Froedtert", "Non-Froedtert"))))))

#this might not work but trying for graphing
HSP.xgb.w.ser$local <- (ifelse(HSP.xgb.w.ser$Provider_ID == '520177', "FH", 
                               (ifelse(HSP.xgb.w.ser$Provider_ID == '520103', "FH-CMH", 
                                       (ifelse(HSP.xgb.w.ser$Provider_ID == '520063', "FH-SJH",
                                               (ifelse(HSP.xgb.w.ser$Provider_ID == '520138', "Aurora St Lukes",
                                                       (ifelse(HSP.xgb.w.ser$Provider_ID == '520139', "Aurora West Allis",
                                                               (ifelse(HSP.xgb.w.ser$Provider_ID == '520051', "Columbia St Marys", "Non-Local"))))))))))))

HSP.xgb.w.ser$Highvalue <- (ifelse(HSP.xgb.w.ser$Highvalue == '1', "High Value", "Non-Value"))

#CSV for to excel
#HSP.SER.CSV <- HSP.xgb.w.ser %>% left_join(serdata.sub.xgb, by=c("Provider_ID"))
#write.csv(HSP.SER.CSV, file = "HSPCER.csv")

#########################################
#                                       #
#      XGB Split violin plotting        #
#                                       #
#########################################

top10features <- subset(HSP.xgb.w.ser, select = c(AVG_SPNDG_PER_EP_HOSPITAL_Carrier_Before,
                                                  AVG_SPNDG_PER_EP_HOSPITAL_Hospice_After,
                                                  AVG_SPNDG_PER_EP_HOSPITAL_IP_After,
                                                  AVG_SPNDG_PER_EP_HOSPITAL_Total_Total,
                                                  PERCENT_OF_SPNDG_HOSPITAL_IP_During,
                                                  AVG_SPNDG_PER_EP_HOSPITAL_SkilledRN_After,
                                                  AVG_SPNDG_PER_EP_HOSPITAL_OP_After,
                                                  AVG_SPNDG_PER_EP_HOSPITAL_Carrier_During,
                                                  PERCENT_OF_SPNDG_HOSPITAL_Carrier_During,
                                                  AVG_SPNDG_PER_EP_HOSPITAL_IP_During,
                                                  Highvalue, FH))

colnames(top10features)[colnames(top10features)=="AVG_SPNDG_PER_EP_HOSPITAL_Carrier_Before"] <- "Avg spending per carrier claim before admission"
colnames(top10features)[colnames(top10features)=="AVG_SPNDG_PER_EP_HOSPITAL_Hospice_After"] <- "Avg spending per hospice claim after admission"
colnames(top10features)[colnames(top10features)=="AVG_SPNDG_PER_EP_HOSPITAL_IP_After"] <- "Avg ep spending per IP claim after admission"
colnames(top10features)[colnames(top10features)=="AVG_SPNDG_PER_EP_HOSPITAL_Total_Total"] <- "Avg episode spending total"
colnames(top10features)[colnames(top10features)=="PERCENT_OF_SPNDG_HOSPITAL_IP_During"] <- "Percent of spending on IP during admission"
colnames(top10features)[colnames(top10features)=="AVG_SPNDG_PER_EP_HOSPITAL_SkilledRN_After"] <- "Avg spending per skilled RN claim after admission"
colnames(top10features)[colnames(top10features)=="AVG_SPNDG_PER_EP_HOSPITAL_OP_After"] <- "Avg spending per OP claim after admission"
colnames(top10features)[colnames(top10features)=="AVG_SPNDG_PER_EP_HOSPITAL_Carrier_During"] <- "Avg spending per carrier claim during admission"
colnames(top10features)[colnames(top10features)=="PERCENT_OF_SPNDG_HOSPITAL_Carrier_During"] <- "Percent of spending per carrier claim during admission"
colnames(top10features)[colnames(top10features)=="AVG_SPNDG_PER_EP_HOSPITAL_IP_During"] <- "Avg ep spending per IP claim during admission"

#hsp.xgb.box <- top10features %>% gather(-Highvalue, key = "var", value = "value") %>% ggplot(aes(x=as.factor(Highvalue), y = value)) + geom_boxplot(width=0.1) +facet_wrap(~ var, scales = "free")

hsp.xgb.violin <- top10features %>% gather(-Highvalue, -FH, key = "var", value = "value") %>% 
                                    ggplot(aes(x=as.factor(Highvalue), y = value, fill = FH)) + 
                                    geom_split_violin() + 
                                    geom_boxplot(width=0.1) + 
                                    facet_wrap(~ var, scales = "free")

hsp.xgb.violin

hsp.xgb.violin.fh <- top10features %>% gather(-Highvalue, -FH, key = "var", value = "value") %>% 
                                       ggplot(aes(x=as.factor(FH), y = value, fill = Highvalue)) + 
                                       geom_split_violin() + 
                                       geom_boxplot(width=0.1) + 
                                       facet_wrap(~ var, scales = "free")

hsp.xgb.violin.fh
#ggplot(top10features, aes(x=as.factor(Highvalue), y = AVG_SPNDG_PER_EP_HOSPITAL_Carrier_Before, fill = FH)) + geom_split_violin()


#########################################
#                                       #
#       GLM Split violin plotting       #
#                                       #
#########################################


glm.top10features <- subset(HSP.xgb.w.ser, select = c(AVG_SPNDG_PER_EP_HOSPITAL_SkilledRN_After,
                                                      PERCENT_OF_SPNDG_HOSPITAL_IP_During,
                                                      PERCENT_OF_SPNDG_HOSPITAL_IP_After,
                                                      PERCENT_OF_SPNDG_HOSPITAL_SkilledRN_Before,
                                                      PERCENT_OF_SPNDG_HOSPITAL_Carrier_After,
                                                      AVG_SPNDG_PER_EP_HOSPITAL_Hospice_After,
                                                      PERCENT_OF_SPNDG_HOSPITAL_Carrier_Before,
                                                      AVG_SPNDG_PER_EP_HOSPITAL_DME_Before,
                                                      AVG_SPNDG_PER_EP_HOSPITAL_IP_Before, 
                                                      PERCENT_OF_SPNDG_HOSPITAL_HomeHealth_Before,
                                                      Highvalue, FH))

colnames(glm.top10features)[colnames(glm.top10features)=="AVG_SPNDG_PER_EP_HOSPITAL_SkilledRN_After"] <- "Avg spending per skilled RN claim after admission"
colnames(glm.top10features)[colnames(glm.top10features)=="PERCENT_OF_SPNDG_HOSPITAL_IP_During"] <- "Percent of spending on IP during admission"
colnames(glm.top10features)[colnames(glm.top10features)=="PERCENT_OF_SPNDG_HOSPITAL_IP_After"] <- "Percent spending per IP claim after admission"
colnames(glm.top10features)[colnames(glm.top10features)=="PERCENT_OF_SPNDG_HOSPITAL_SkilledRN_Before"] <- "Percent spending per skilled RN claim before admission"
colnames(glm.top10features)[colnames(glm.top10features)=="PERCENT_OF_SPNDG_HOSPITAL_Carrier_After"] <- "Percent spending per carrier claim After admission"
colnames(glm.top10features)[colnames(glm.top10features)=="AVG_SPNDG_PER_EP_HOSPITAL_Hospice_After"] <- "Avg spending per hospice claim after admission"
colnames(glm.top10features)[colnames(glm.top10features)=="PERCENT_OF_SPNDG_HOSPITAL_Carrier_Before"] <- "Percent spending per carrier claim before admission"
colnames(glm.top10features)[colnames(glm.top10features)=="AVG_SPNDG_PER_EP_HOSPITAL_DME_Before"] <- "Avg spending per DME claim before admission"
colnames(glm.top10features)[colnames(glm.top10features)=="AVG_SPNDG_PER_EP_HOSPITAL_IP_Before"] <- "Avg spending per IP claim before admission"
colnames(glm.top10features)[colnames(glm.top10features)=="PERCENT_OF_SPNDG_HOSPITAL_HomeHealth_Before"] <- "Percent spending per HomeHealth claim before admission"


hsp.glm.violin <- glm.top10features %>% gather(-Highvalue, -FH, key = "var", value = "value") %>% 
  ggplot(aes(x=as.factor(Highvalue), y = value, fill = FH)) + 
  geom_split_violin() + 
  geom_boxplot(width=0.1) + 
  facet_wrap(~ var, scales = "free") + 
  scale_fill_manual(values = c("#094897","#398c1d"))

plot(hsp.glm.violin)

hsp.glm.bar.fh <- ggplot(data = glm.top10features %>% gather(-Highvalue, -FH, key = "var", value = "value"), 
                         aes(x=as.factor(Highvalue), y = value, fill = FH)) + 
                         geom_bar(position = "dodge", stat = "summary", fun.y = "mean") +
                         stat_summary(aes(label=round(..y..,2)), fun.y=mean, geom="text", size=3,vjust = -0.5) + 
                         facet_wrap(~ var, scales = "free") + scale_fill_manual(values = c("#094897","#398c1d"))

plot(hsp.glm.bar.fh)


#hsp.glm.bar.local <- ggplot(data = glm.top10features %>% gather(-Highvalue, -local, key = "var", value = "value"), 
#                      aes(x=as.factor(Highvalue), y = value, fill = local)) + 
#                      geom_bar(position = "dodge", stat = "summary", fun.y = "mean") +
#                      facet_wrap(~ var, scales = "free")

#plot(hsp.glm.bar.local)




#########################################
#                                       #
#       Putting the Story Together      #
#                                       #
#########################################



glm.export <- subset(HSP.xgb.w.ser, select = c(AVG_SPNDG_PER_EP_HOSPITAL_SkilledRN_After,
                                               PERCENT_OF_SPNDG_HOSPITAL_IP_During,
                                               PERCENT_OF_SPNDG_HOSPITAL_IP_After,
                                               PERCENT_OF_SPNDG_HOSPITAL_SkilledRN_Before,
                                               PERCENT_OF_SPNDG_HOSPITAL_Carrier_After,
                                               AVG_SPNDG_PER_EP_HOSPITAL_Hospice_After,
                                               PERCENT_OF_SPNDG_HOSPITAL_Carrier_Before,
                                               AVG_SPNDG_PER_EP_HOSPITAL_DME_Before,
                                               AVG_SPNDG_PER_EP_HOSPITAL_IP_Before, 
                                               PERCENT_OF_SPNDG_HOSPITAL_HomeHealth_Before,
                                               Highvalue, FH, local, Provider_ID))

colnames(glm.export)[colnames(glm.export)=="AVG_SPNDG_PER_EP_HOSPITAL_SkilledRN_After"] <- "Avg spending per skilled RN claim after admission"
colnames(glm.export)[colnames(glm.export)=="PERCENT_OF_SPNDG_HOSPITAL_IP_During"] <- "Percent of spending on IP during admission"
colnames(glm.export)[colnames(glm.export)=="PERCENT_OF_SPNDG_HOSPITAL_IP_After"] <- "Percent spending per IP claim after admission"
colnames(glm.export)[colnames(glm.export)=="PERCENT_OF_SPNDG_HOSPITAL_SkilledRN_Before"] <- "Percent spending per skilled RN claim before admission"
colnames(glm.export)[colnames(glm.export)=="PERCENT_OF_SPNDG_HOSPITAL_Carrier_After"] <- "Percent spending per carrier claim After admission"
colnames(glm.export)[colnames(glm.export)=="AVG_SPNDG_PER_EP_HOSPITAL_Hospice_After"] <- "Avg spending per hospice claim after admission"
colnames(glm.export)[colnames(glm.export)=="PERCENT_OF_SPNDG_HOSPITAL_Carrier_Before"] <- "Percent spending per carrier claim before admission"
colnames(glm.export)[colnames(glm.export)=="AVG_SPNDG_PER_EP_HOSPITAL_DME_Before"] <- "Avg spending per DME claim before admission"
colnames(glm.export)[colnames(glm.export)=="AVG_SPNDG_PER_EP_HOSPITAL_IP_Before"] <- "Avg spending per IP claim before admission"
colnames(glm.export)[colnames(glm.export)=="PERCENT_OF_SPNDG_HOSPITAL_HomeHealth_Before"] <- "Percent spending per HomeHealth claim before admission"

DRG.HSP.SER.Export <- drg.ser %>% left_join(glm.export, by=c("Provider_ID"))

# Difference between AMP and ATP (ATP - AMP) is the patient burden/responsibility 
DRG.HSP.SER.Export$patientburden <- ((DRG.HSP.SER.Export$ATPperDischarge - DRG.HSP.SER.Export$AMPperDischarge))
#DRG.HSP.SER.Export$PercentCMSperDischarge <- format(round(drgdata$PercentCMSperDischarge, 2), nsmall = 2)


#Filter MDC's for those which match the SER/Provdier episode spending presumed MDC's
#This does limit the analysis of which drgs have the highest patient burden
#however, it allows us to make a more accurate inference as to the value spending 
#when using the provider value and hospital episode data sets. 
DRG.HSP.SER.MDCsub <- DRG.HSP.SER.Export %>% filter(MDC %in% c('05','04','08'))

high.value.subset <- DRG.HSP.SER.MDCsub %>% filter(Highvalue %in% "High Value")
non.value.subset <- DRG.HSP.SER.MDCsub %>% filter(Highvalue %in% "Non-Value")
fh.value.subset <- DRG.HSP.SER.MDCsub %>% filter(FH.y %in% "Froedtert")


#Averaging the patient burden by DRG for high value and non-value patietns (column 41 is pt burden, update to name when there is time)
high.value.drg.avg <- aggregate(high.value.subset[, 40], list(high.value.subset$DRGDefinition), mean)
non.value.drg.avg <- aggregate(non.value.subset[, 40], list(non.value.subset$DRGDefinition), mean)


colnames(high.value.drg.avg)[colnames(high.value.drg.avg)=="patientburden"] <- "High Value Patient Burden"
colnames(non.value.drg.avg)[colnames(non.value.drg.avg)=="patientburden"] <- "Non-value patient burden"

high.non.value.drg <-merge(high.value.drg.avg,non.value.drg.avg)

high.non.value.drg$delta <- (high.non.value.drg$`High Value Patient Burden` - high.non.value.drg$`Non-value patient burden`)
high.non.value.drg$Percent_Delta <- (((high.non.value.drg$`High Value Patient Burden` - high.non.value.drg$`Non-value patient burden`)/ high.non.value.drg$`High Value Patient Burden`) * 100)

#parse DRG from title
high.non.value.drg$drgID <- substr(high.non.value.drg$Group.1, start = 1, stop = 3)

#Map DRG to MDC
high.non.value.drg$MDC <- DRGtoMDC$MDC[match(high.non.value.drg$drgID, DRGtoMDC$`MS-DRG`)]

#Try to find the DRG with the biggest difference in patient burden between FH and Highvalue providers
high.FH.value.subset <- DRG.HSP.SER.MDCsub %>% filter(Highvalue %in% "High Value" | FH.y %in% "Froedtert")
colnames(high.value.drg.avg)[colnames(high.value.drg.avg)=="Group.1"] <- "DRGDefinition"

#Join the high value average patient burden per DRG to the FH only DRG's then calculate the delta between avg high value and FH
fh.high.wide <- fh.value.subset %>% left_join(high.value.drg.avg, by=c("DRGDefinition"))
fh.high.wide$delta <- (fh.high.wide$`High Value Patient Burden` - fh.high.wide$patientburden)
fh.high.wide$Percent_Delta <- (((fh.high.wide$`High Value Patient Burden` - fh.high.wide$patientburden) / fh.high.wide$`High Value Patient Burden`) * 100)

#
# ----------------------------------------------------------------------------------This PLOT IS EVIL, DESTROY
#

#Plot all FH DRG percent delta from high value providers average patient burdent per DRG
# fh.high.wide.delta.plot <- ggplot(data=fh.high.wide,
#                                   aes(x=DRGDefinition, y=Percent_Delta, group = 1)) + 
#                                   theme(axis.text.x = element_text(angle = 45)) +
#                                   geom_jitter() + 
#                                   geom_smooth(method = "lm", show.legend = TRUE) + 
#                                   ylim(-750,50) + #scale_y_continuous() +
#                                   labs(title="Percent Decrease in Patient Burden per DRG", 
#                                        subtitle="Difference Between Froedtert Patient Burden per DRG and High Value Providers (National AVG)",
#                                        y="Percent Decrease in Patient Burden", 
#                                        x="DRG", 
#                                        caption="Band Represents 95% Confidence Interval")
# 
# plot(fh.high.wide.delta.plot)


#column 40 is patient burden
fh.high.wide.mdc.avg <- aggregate(fh.high.wide[, 40], list(fh.high.wide$MDC), mean)
#column 41 is high value patient burden
fh.high.wide.mdc.avg$highvalueptburden <- aggregate(fh.high.wide[, 41], list(fh.high.wide$MDC), mean, na.action = na.omit)

high.mdc.box <- fh.high.wide %>% select(MDC, `High Value Patient Burden`) %>% 
                                 ggplot(aes(x=MDC, y=`High Value Patient Burden`)) + 
                                 labs(title="High Value and Patient Burden by MDC") +
                                 geom_boxplot() +
                                 theme(plot.title=element_text(hjust=0.5, color="white"), plot.background=element_rect(fill="#808484"), 
                                        axis.text.x=element_text(colour="white"), axis.text.y=element_text(colour="white"), 
                                        axis.title=element_text(colour="white"))

fh.mdc.box <- fh.high.wide %>% select(MDC, patientburden) %>% 
                               ggplot(aes(x=MDC, y=patientburden)) + 
                               labs(title="FH Patient Burden by MDC") +
                               geom_boxplot() +
                               theme(plot.title=element_text(hjust=0.5, color="white"), plot.background=element_rect(fill="#808484"), 
                                    axis.text.x=element_text(colour="white"), axis.text.y=element_text(colour="white"), 
                                    axis.title=element_text(colour="white"))

grid.arrange(high.mdc.box,fh.mdc.box, ncol = 2, nrow = 1)


high.fh.mdc.box <- ggplot(fh.high.wide, aes(MDC)) +                    
                          geom_violin(aes(y=`High Value Patient Burden`), colour="blue") + 
                          geom_violin(aes(y=patientburden), colour="red", alpha = 0.5) + 
                          labs(title="Patient Cost Burden by MDC for FH and High Value Providers", 
                               subtitle="All High Value Providers as Average per MDC",
                               y="Patient Burden", 
                               x="Major Diagnostic Category", 
                               caption="Blue: High Value Providers, Red: FH") 
#scale_color_manual(name="Provider", 
#                                              labels = c("High Value (avg)", 
#                                                         "Froedtert"), 
#                                              values = c("High Value (avg)"="blue", 
#                                                         "Froedtert" = "red"))
high.fh.mdc.box


#########################################
#                                       #
#       Final Graph...graphics          #
#                                       #
#########################################


melt.hfhvs <- melt(fh.high.wide, id=c("DRGDefinition"), measure.vars=c("patientburden", "High Value Patient Burden"), variable.name="Provider",value.name="patientburden")
melt.hfhvs$Name <- (ifelse(melt.hfhvs$Provider %in% "patientburden", 'Froedtert',
                           ifelse(melt.hfhvs$Provider %in% "High Value Patient Burden", 'High Value', '')))
melt.hfhvs$drgID <- substr(melt.hfhvs$DRGDefinition, start = 1, stop = 3)
melt.hfhvs$MDC <- DRGtoMDC$MDC[match(melt.hfhvs$drgID, DRGtoMDC$`MS-DRG`)]


#
# ------------------------REPLACe WITH BAR GRAPH
#


melt.hfhvs.plot <- melt.hfhvs %>% select(drgID, Name, patientburden) %>% 
                                  ggplot(aes(x=drgID, y=patientburden ,color = Name)) + 
                                  theme(axis.text.x = element_text(angle = 45)) +
                                  geom_jitter(shape=21) + 
                                  scale_color_manual(values = c("#094897","#398c1d")) + 
                                  scale_y_log10() +
                                  labs(title="Patient Burden per DRG", 
                                       subtitle="Froedtert and High Value Proviers (National AVG)",
                                       y="Patient Burden", 
                                       x="DRG")
# donot include
plot(melt.hfhvs.plot)

g1 <- ggplot(melt.hfhvs, aes(x = patientburden, fill = Name)) + 
              geom_density(alpha = 0.7) + 
              scale_fill_manual(values = c("#094897","#398c1d")) + 
              labs(x="Patient Burden")

g2 <- ggplot(melt.hfhvs, aes(x = MDC, y = patientburden, fill = Name)) + 
              geom_violin(alpha = 0.7) + geom_boxplot(width=0.1) + 
              scale_fill_manual(values = c("#094897","#398c1d")) + 
              labs(x="Major Disease Category", y="Patient Burden")


melt.hfhvs.plot.bar <- melt.hfhvs %>% select(drgID, Name, patientburden) %>% 
                        ggplot(aes(x=drgID, y=patientburden)) + 
                        theme(axis.text.x = element_text(angle = 45)) +
                        geom_col(aes(fill = Name)) + 
                        scale_fill_manual(values = c("#094897","#398c1d")) +
                        labs(title="Patient Burden per DRG", 
                             subtitle="Froedtert and High Value Proviers (National AVG)", 
                             y="Patient Burden", 
                             x="DRG")



g1
g2
grid.arrange(melt.hfhvs.plot.bar, g1, g2, widths = c(4, 2, 2),
             layout_matrix = rbind(c(1, 2),
                                   c(1, 3)))


#########################################
#                                       #
#       Final Bar Plots                 #
#                                       #
#########################################

glm.bar.final <- ggplot(data = glm.top10features %>% gather(-Highvalue, -FH, key = "var", value = "value"), 
                        aes(x=as.factor(Highvalue), y = value, fill = FH)) + 
                        geom_bar(position = "dodge", stat = "summary", fun.y = "mean") +
                        stat_summary(aes(label=round(..y..,2)), fun.y=mean, geom="text", size=3, vjust = -0.5) + 
                        facet_wrap(~ var, scales = "free") + 
                        theme(axis.text.x = element_text(angle = 45)) + 
                        scale_fill_manual(values = c("#094897","#398c1d")) 

plot(glm.bar.final)

#Creating Dataframe of just FH and High Value providers for closer analysis
glm.bar.data <- glm.top10features
glm.bar.data$HVFH <- (ifelse(grepl("High Value", glm.bar.data$Highvalue), 'High Value',
                             ifelse(glm.bar.data$FH %in% "Froedtert", 'Froedtert', "0")))
glm.bar.HVFH <- glm.bar.data %>% filter(HVFH != "0")



bar1 <- ggplot(data=glm.bar.HVFH, aes(x=HVFH, y=`Avg spending per skilled RN claim after admission`, fill = HVFH)) + 
              geom_bar(stat="summary", fun.y = "mean", position=position_dodge()) + 
              scale_fill_manual(values = c("#094897","#398c1d")) + 
              theme(legend.position = "none", axis.text.x = element_text(angle = 45)) + 
              stat_summary(geom = "errorbar", fun.data = mean_se, position = "dodge", width=.2) +
              labs(title="Skilled RN Claim After Admission", 
                   subtitle="Froedtert and High Value Proviers (National AVG)",
                   y="Average Spending", 
                   x="", 
                   caption = "1 through 30 days After Discharge from Index Hospital \n Admission, Skilled Nursing Facility Claim Type ")

bar2 <- ggplot(data=glm.bar.HVFH, aes(x=HVFH, y=`Percent of spending on IP during admission`, fill = HVFH)) + 
              geom_bar(stat="summary", fun.y = "mean", position=position_dodge()) +
              scale_fill_manual(values = c("#094897","#398c1d")) +  
              theme(legend.position = "none", axis.text.x = element_text(angle = 45)) +
              stat_summary(geom = "errorbar", fun.data = mean_se, position = "dodge", width=.2) +
              labs(title="IP During Admission", 
                   subtitle="Froedtert and High Value Proviers (National AVG)",
                   y="Percent of Spending", 
                   x="", 
                   caption = "During Index Hospital Admission,\n Inpatient Claim Type")

bar3 <- ggplot(data=glm.bar.HVFH, aes(x=HVFH, y=`Percent spending per IP claim after admission`, fill = HVFH)) + 
              geom_bar(stat="summary", fun.y = "mean", position=position_dodge()) +
              scale_fill_manual(values = c("#094897","#398c1d")) +  
              theme(legend.position = "none", axis.text.x = element_text(angle = 45)) +
              stat_summary(geom = "errorbar", fun.data = mean_se, position = "dodge", width=.2) +
              labs(title="IP Claim After Admission", 
                   subtitle="Froedtert and High Value Proviers (National AVG)",
                   y="Percent of Spending", 
                   x="", 
                   caption = "1 through 30 days After Discharge from Index Hospital \n Admission, Inpatient Claim Type")

#bar4 <- ggplot(data=glm.bar.HVFH, aes(x=HVFH, y=`Percent spending per skilled RN claim before admission`, fill = HVFH)) + geom_bar(stat="summary", fun.y = "mean", position=position_dodge()) +
#        scale_fill_manual(values = c("#094897","#398c1d")) +  theme(legend.position = "none", axis.text.x = element_text(angle = 45)) +
#        stat_summary(geom = "errorbar", fun.data = mean_se, position = "dodge", width=.2) +
#        labs(title="Percent Spending per Skilled RN Claim Before Admission", subtitle="Froedtert and High Value Proviers (National AVG)",
#        y="Percent of Spending", x="")

bar5 <- ggplot(data=glm.bar.HVFH, aes(x=HVFH, y=`Percent spending per carrier claim After admission`, fill = HVFH)) + 
              geom_bar(stat="summary", fun.y = "mean", position=position_dodge()) +
              scale_fill_manual(values = c("#094897","#398c1d")) +  
              theme(legend.position = "none", axis.text.x = element_text(angle = 45)) +
              stat_summary(geom = "errorbar", fun.data = mean_se, position = "dodge", width=.2) +
              labs(title="Carrier Claim After Admission", 
                   subtitle="Froedtert and High Value Proviers (National AVG)",
                   y="Percent of Spending", 
                   x="", 
                   caption = "1 through 30 days After Discharge from Index Hospital Admission,\n Carrier (Ambulatory and Ancillary Services) Claim Type")

bar6 <- ggplot(data=glm.bar.HVFH, aes(x=HVFH, y=`Avg spending per hospice claim after admission`, fill = HVFH)) + 
              geom_bar(stat="summary", fun.y = "mean", position=position_dodge()) +
              scale_fill_manual(values = c("#094897","#398c1d")) +  
              theme(legend.position = "none", axis.text.x = element_text(angle = 45)) +
              stat_summary(geom = "errorbar", fun.data = mean_se, position = "dodge", width=.2) +
              labs(title="Hospice Claim After Admission", 
                   subtitle="Froedtert and High Value Proviers (National AVG)",
                   y="Average Spending", 
                   x="", 
                   caption = "1 through 30 days After Discharge from Index Hospital \n Admission, Hospice Claim Type")


grid.arrange(bar1,bar2,bar3,bar5,bar6, ncol=3, nrow=2)

bar1
bar2
bar3
bar5
bar6
#########################################
#                                       #
#     Does Index admission spending     #
#     impact after discharge spending?
#     Actually, no not really... 
#                                       #
#########################################


IP.After.spending <- HSP.xgb.w.ser
write.csv(IP.After.spending, file = "IP_After.csv")



#high.value.drg.avg <- aggregate(high.value.subset[, 41], list(high.value.subset$DRGDefinition), mean)
#non.value.drg.avg <- aggregate(non.value.subset[, 41], list(non.value.subset$DRGDefinition), mean)
#fh.high.wide$BurdenDelta <- (fh.high.wide$patientburden.y - fh.high.wide$patientburden.x)
#fh.high.burdenbyDRG <- fh.high.wide[c(1,14,24,26,82)]
#rm(fh.high.wide,fh.high.burdenbyDRG)

#########################################
#                                       #
#               LANDFILL                #
#                                       #
#########################################
#
#
#
#Quality Rank score Table
#	#	#	#	#	#	#	#
#	#	#	#	#	#	#	#
#	Not Available	Rank 0-10 high = 10		0-10 high = 10	#	AVG score: 	#
#	Average mortality 	5	 average payment	5	#	5	#
#	Average mortality 	5	 higher payment	0	#	2.5	#
#	Average complications 	5	 average payment	5	#	5	#
#	Average mortality 	5	 lower payment	10	#	7.5	#
#	Worse complications 	0	 higher payment	0	#	0	#
#	Worse mortality 	0	 average payment	5	#	2.5	#
#	Better mortality 	10	 higher payment	0	#	5	#
#	Average complications 	5	 higher payment	0	#	2.5	#
#	Average complications 	5	 lower payment	10	#	7.5	#
#	Worse mortality 	0	 higher payment	0	#	0	#
#	Better mortality 	10	 average payment	5	#	7.5	#
#	Worse mortality 	0	 lower payment	10	#	5	#
#	Better mortality 	10	 lower payment	10	#	10	#
#	Better complications 	10	 lower payment	10	#	10	#
#	Worse complications 	0	 lower payment	10	#	5	#
#	Better complications 	10	 average payment	5	#	7.5	#
#	Worse complications 	0	 average payment	5	#	2.5	#
#	Better complications 	10	 higher payment	0	#	5	#
#	#	#	#	#	#	#	#





