select i.itemid, l.count 
from procedure_events i
join procedure_events_icu_count l 
on l.itemid = i.itemid 
group by i.itemid, l.count
order by i.itemid