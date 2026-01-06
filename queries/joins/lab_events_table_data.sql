select i.itemid, i.label, i.fluid, i.category, l.count 
from lab_items i
join lab_events_icu_count l 
on l.itemid = i.itemid 
group by i.itemid, i.label, i.fluid, i.category, l.count
order by i.itemid