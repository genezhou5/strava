# Strava

library("ggplot2")
library("tidyverse")
library('readr')
library('lubridate')
library('stringr')
library('clock')
library('gridExtra')

setwd("C:\\Users\\gzhou\\Documents\\Strava\\Data")

activities <- read_csv("activities.csv")

data <- activities %>%
  select(`Activity Date`, `Activity Type`, Distance, `Elapsed Time`, `Perceived Exertion`,
         `Elevation Gain`, `Relative Effort`) %>%
  mutate(activity_date = with_tz(parse_date_time(`Activity Date`, orders='mdY HMS p', tz='UTC'),
                                 'America/Los_Angeles'))%>%
  mutate(elapsed_time_minutes = `Elapsed Time`/60,
         distance_mi = Distance*0.62137119) %>%
  mutate(activity_week = as.Date(cut(activity_date, "week"))) %>%
  rename(activity_type = `Activity Type`,
         distance_km = Distance,
         elapsed_time = `Elapsed Time`,
         perceived_exertion = `Perceived Exertion`,
         elevation_gain = `Elevation Gain`,
         relative_effort = `Relative Effort`) %>%
  select(activity_date, activity_week, activity_type, distance_km, distance_mi,
         elapsed_time_minutes, perceived_exertion, elevation_gain, relative_effort)


runs <- data %>% filter(activity_type == 'Run' &
                          activity_date >= ymd('2022-02-20')) %>%
  mutate(pace = elapsed_time_minutes/distance_mi)

# Run Data
runs_weekly <- runs %>%
  group_by(activity_week) %>%
  summarise(distance_km = sum(distance_km),
            distance_mi = sum(distance_mi),
            time_minutes = sum(elapsed_time_minutes),
            elevation_gain = sum(elevation_gain))

run_mi_plt <- ggplot(runs_weekly, aes(activity_week, distance_mi, group=1)) +
  geom_line() +
  xlab('') +
  ylab('Distance (mi)') +
  ggtitle('Weekly Mileage') +
  theme(axis.text.x=element_text(angle=45, hjust=1),
        plot.title=element_text(hjust=0.5))

run_tm_plt <- ggplot(runs_weekly, aes(activity_week, time_minutes, group=1)) +
  geom_line() +
  xlab('') +
  ylab('Time (min)') +
  ggtitle('Weekly Time Running') +
  theme(axis.text.x=element_text(angle=45, hjust=1),
        plot.title=element_text(hjust=0.5))

run_pc_plt <- ggplot(runs, aes(activity_week, pace)) +
  geom_smooth(span=0.1, se=TRUE) +
  #geom_line() +
  xlab('Month') +
  ylab('Pace (min/mi)') +
  ggtitle('Avg Pace') +
  theme(axis.text.x=element_text(angle=45, hjust=1),
        plot.title=element_text(hjust=0.5))

run_el_plt <- ggplot(runs_weekly, aes(activity_week, elevation_gain, group=1)) +
  geom_line() +
  xlab('') +
  ylab('Elevation Gain (m)') +
  ggtitle('Elevation Gain') +
  theme(axis.text.x=element_text(angle=45, hjust=1),
        plot.title=element_text(hjust=0.5))

# Swim Data
swims <- data %>% filter(activity_type == 'Swim' &
                          activity_date >= ymd('2022-02-20')) %>%
  rename(distance_m = distance_km) %>%
  mutate(distance_mi = distance_m/1609) %>%
  select(-elevation_gain)

swims_weekly <- swims %>%
  group_by(activity_week) %>%
  summarise(distance_m = sum(distance_m),
            distance_mi = sum(distance_mi),
            time_minutes = sum(elapsed_time_minutes))
  
swim_m_plt <- ggplot(swims_weekly, aes(activity_week, distance_m, group=1)) +
  geom_line() +
  xlab('') +
  ylab('Distance (m)') +
  ggtitle('Weekly Swim Distance') +
  theme(axis.text.x=element_text(angle=45, hjust=1),
        plot.title=element_text(hjust=0.5)) 

grid.arrange(run_mi_plt, run_tm_plt, run_pc_plt, swim_m_plt, ncol=1)

###########################
# TEST CODE PLEASE IGNORE #
###########################

## Need to account for am/pm
date_tapi <- activities %>%
  select(`Activity Date`) %>%
  mutate(utc = parse_date_time(`Activity Date`,
                               orders='mdY HMS p',
                               tz='UTC')) %>%
  mutate(pst = with_tz(utc, 'America/Los_Angeles'))


