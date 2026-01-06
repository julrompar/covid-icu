select i.itemid, l.count 
from chart_events_icu i
join chart_events_icu_count l 
on l.itemid = i.itemid 
group by i.itemid, l.count
order by i.itemid