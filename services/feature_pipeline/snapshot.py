query_snapshot = """
-- CREATE TABLE train_snapshot AS
with sales_snapshot as (
  select 
    sa_date as dt, 
    sa_p_id as pid, 
    sa_s_id as sid,
    es_e_id as eid,
    p_group as pg,
    not (pr_id is null) as iop,
    ln(sa_units_sold + 1) as lus
  from core.sales
  left join core.eventstores on
    sa_date = es_date and sa_s_id = es_s_id
  left join promotions on
    sa_date = pr_date and sa_p_id = pr_p_id and sa_s_id = pr_s_id
  left join products on
    sa_p_id = p_id
),
lag_sales as (
  select 
    c_date + 1 as l1d_dt,
    c_date + 2 as l2d_dt,
    product_id,
    store_id,
    units_sold
  from sales_snapshot
),
next_sales as (
  select
    pid,
    sid,
    iop,
    eid,
    lus,
    dt - 1 as n1d_dt,
    dt - 2 as n2d_dt,
    dt - 3 as n3d_dt,
    dt - 4 as n4d_dt,
    dt - 5 as n5d_dt,
    dt - 6 as n6d_dt,
    dt - 7 as n7d_dt,
  from sales_snapshot as ss
),
sales_featured as (
  select 
    ss.dt as c_date,
    ss.pid as product_id,
    ss.sid as store_id,
    ss.lus as log_units_sold,
    ss.pg as product_group,
    n1d.eid as next_1d_event_id,
    n2d.eid as next_2d_event_id,
    n3d.eid as next_3d_event_id,
    n4d.eid as next_4d_event_id,
    n5d.eid as next_5d_event_id,
    n6d.eid as next_6d_event_id,
    n7d.eid as next_7d_event_id,
    coalesce(n1d.iop, false) as next_1d_is_on_promo,
    coalesce(n2d.iop, false) as next_2d_is_on_promo,
    coalesce(n3d.iop, false) as next_3d_is_on_promo,
    coalesce(n4d.iop, false) as next_4d_is_on_promo,
    coalesce(n5d.iop, false) as next_5d_is_on_promo,
    coalesce(n6d.iop, false) as next_6d_is_on_promo,
    coalesce(n7d.iop, false) as next_7d_is_on_promo,
    n1d.lus as h1_log_units_sold,
    n2d.lus as h2_log_units_sold, 
    n3d.lus as h3_log_units_sold, 
    n4d.lus as h4_log_units_sold,
    n5d.lus as h5_log_units_sold,
    n6d.lus as h6_log_units_sold, 
    n7d.lus as h7_log_units_sold, 
    -- l1d.units_sold,
    -- l2d.units_sold
  from sales_snapshot as ss
  left join next_sales as n1d on 
    n1d.n1d_dt = dt and n1d.pid = ss.pid and n1d.sid = ss.sid
  left join next_sales as n2d on 
    n2d.n2d_dt = dt and n2d.pid = ss.pid and n2d.sid = ss.sid
  left join next_sales as n3d on 
    n3d.n3d_dt = dt and n3d.pid = ss.pid and n3d.sid = ss.sid
  left join next_sales as n4d on 
    n4d.n4d_dt = dt and n4d.pid = ss.pid and n4d.sid = ss.sid
  left join next_sales as n5d on 
    n5d.n5d_dt = dt and n5d.pid = ss.pid and n5d.sid = ss.sid
  left join next_sales as n6d on 
    n6d.n6d_dt = dt and n6d.pid = ss.pid and n6d.sid = ss.sid
  left join next_sales as n7d on 
    n7d.n7d_dt = dt and n7d.pid = ss.pid and n7d.sid = ss.sid
  -- left join lag_sales as l1d on l1d_dt = dt and l1d.pid = ss.pid and l1d.sid = ss.sid
  -- left join lag_sales as l2d on l2d_dt = dt and l2d.pid = ss.pid and l2d.sid = ss.sid
)
select *
from sales_featured
where 1=1
  and c_date between '{date_from}' and '{date_upto}'
order by product_id, store_id, c_date;
"""
