library(arules);
data(Titanic);

df <- as.data.frame(Titanic)
titanic.raw <- NULL
for (i in 1:4)
{
  titanic.raw <- cbind(titanic.raw, rep(as.character(df[,i]), df$Freq));
}
titanic.raw <- as.data.frame(titanic.raw);
names(titanic.raw) <- names(df)[1:4];


#rules <- apriori(Groceries, parameter = list(supp = 0.001, conf = 0.001, minlen = 2, maxlen = 2));
rules <- apriori(titanic.raw, parameter = list(supp = .001, conf = .001, minlen = 2), appearance = list(default = "lhs", rhs = "Class=1st"));

# rules with the highest confidence
print("Sorted By Confidence");
rulesConf <- sort(rules, by="confidence", decreasing = TRUE);
inspect(head(rulesConf, 10));

# rules with the highest support
print("Sorted By Support:");
rulesSup <- sort(rules, by="support", decreasing = TRUE);
inspect(head(rulesSup, 10));

print("Sorted By Lift:");
rulesLift <- sort(rules, by="lift", decreasing = TRUE);
inspect(head(rulesLift, 10));



# rules with the highest confidence
#print("Sorted By Confidence");
#rulesConf <- sort(rules, by="confidence", decreasing = TRUE);
#inspect(head(rulesConf, 20));

# rules with the highest support
#print("Sorted By Support:");
#rulesSup <- sort(rules, by="support", decreasing = TRUE);
#inspect(head(rulesSup, 20));

#print("Sorted By Lift:");
#rulesLift <- sort(rules, by="lift", decreasing = TRUE);
#inspect(head(rulesLift, 100));
