rm(list = ls())
library(tidyverse)


loss_df = read.csv("F:\\PycharmProjects\\complex_hierarchy\\data\\Output-2023-06-22\\lstm\\lstm_loss.csv")
loss_df = loss_df %>% mutate(kld.1 = (loss.lang.1-best.lang.1)/log(2)) %>% mutate(kld.2 = (loss.lang.2-best.lang.2)/log(2)) %>% mutate(g1.pref = kld.2-kld.1)
ggplot(loss_df, aes(x = l0, y=g1.pref)) + geom_point()+geom_smooth(size=1, color="red")+geom_rect(xmin=25, xmax=75, ymin=-1,ymax=1,alpha=.007,color="gray")+xlab("Number of Parameters (expected value)")+ylab("G1 Preference")+coord_cartesian(xlim=c(10,200), ylim=c(-.2, .6))+
  theme_bw()+ theme(text = element_text(size=17),legend.position="none")+geom_hline(yintercept = 0, linetype = "dashed", color = "red")+
  geom_segment(x = 30, xend = 30, y = 0, yend = -.15,
               arrow = arrow(length = unit(0.3, "cm")),
               color = "red", size=1) +
  geom_text(x = 60, y = -.05, label = "G3 Preferred",
            hjust = 1, vjust = 0, color = "red", alpha=.75) +
  geom_segment(x = 30, xend = 30, y = 0, yend = .15,
               arrow = arrow(length = unit(0.3, "cm")),
               color = "blue", size=1) +
  geom_text(x = 60, y = .05, label = "G1 Preferred",
            hjust = 1, vjust = 0, color = "blue", alpha=.75)
  
ggplot(data=loss_df, aes(x = l0)) + geom_point(aes(y=kld.1), color="red")+geom_smooth(aes(y=kld.1), color="red")+ geom_point(aes(y=kld.2), color="black")+geom_smooth(aes(y=kld.2), color="black")+geom_rect(xmin=25, xmax=75, ymin=-1,ymax=1,alpha=.007)+xlab("Number of Parameters (expected value)")+ylab("Kullback-Liebler Divergence (bits)")+coord_cartesian(xlim=c(10,200), ylim=c(0, .75))+
  theme_bw()+ theme(text = element_text(size=17),legend.position="none")

ggplot(data=loss_df, aes(x=l0, y=loss)) + geom_point()
