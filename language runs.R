library(tidyverse)

frontiers1 = read.csv("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\Output-2022-11-06\\dyck1_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="dyck")
frontiers2 = read.csv("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\Output-2022-11-06\\abn_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="abn")
frontiers3 = read.csv("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\Output-2022-11-06\\anbn_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="anbn")
frontiers4 = read.csv("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\Output-2022-11-06\\a2nb2m_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="a2nb2m")
frontiers5 = read.csv("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\Output-2022-11-08\\dyck1_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="dyck")
frontiers6 = read.csv("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\Output-2022-11-08\\abn_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="abn")
frontiers7 = read.csv("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\Output-2022-11-08\\anbn_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="anbn")
frontiers8 = read.csv("C:\\Users\\torre\\PycharmProjects\\complex_hierarchy\\Output-2022-11-08\\a2nb2m_small_lstm_loss.csv", header = TRUE, fileEncoding="UTF-8-BOM") %>% mutate(lang="a2nb2m")




#frontiers = rbind(frontiers1, frontiers2)
#frontiers = rbind(frontiers, frontiers3)
frontiers = rbind(frontiers1, frontiers2, frontiers3, frontiers4, frontiers5, frontiers6, frontiers7, frontiers8)%>% mutate(is_regular = ((lang == "a2nb2m")|(lang=="abn")))

frontiers$model_num <- as.factor(frontiers$model_num)
frontiers$lambda <- as.factor(frontiers$lambda)
frontiers_filtered = frontiers #%>% filter(((epoch>1000)|((lang=="abn") & (epoch>500)))&(accuracy>.995))%>% mutate(log_loss = log(loss)) %>% filter(is.finite(log_loss))
frontiers_hulls = frontiers_filtered%>%select(lang,l0,loss,epoch)%>% group_by(lang) %>% slice(chull(l0,log(loss)*32))
plot = ggplot(frontiers%>%filter(epoch>1500), aes(x = l0, y = loss)) + geom_point(aes(color = epoch)) +xlab("Number of parameters")+ylab("Bernoulli Loss (log scale)") + theme_classic() + facet_wrap(~lang)
plot + scale_y_continuous(breaks = c(10e-2, 10e-4, 10e-6, 10e-8), limits = c(10e-9,10e-1), trans="log2")

analysis = manova(cbind(l0, loss) ~ lang, data=frontiers_filtered)
summary(analysis)


frontiers_modified = frontiers %>% mutate(is_regular = ((lang == "a2nb2m")|(lang=="abn")))
t.test(l0~is_regular, data=frontiers%>%filter(epoch>1500))
