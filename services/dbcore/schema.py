from datetime import date
from typing import Optional

from sqlalchemy import CheckConstraint, Sequence, UniqueConstraint
from sqlmodel import Field, SQLModel


def id_field(table_name: str):
    """
    CREATE SEQUENCE IF NOT EXISTS table_name_id_seq START 1;
    """
    sequence = Sequence(f"{table_name}_id_seq")
    return Field(
        default=None,
        primary_key=True,
        unique=True,
        sa_column_args=[sequence],
        sa_column_kwargs={"server_default": sequence.next_value()},
    )


class ProductGroups(SQLModel, table=True):
    """
    CREATE TABLE IF NOT EXISTS product_groups (
        pg1_id VARCHAR PRIMARY KEY,
        pg1_name VARCHAR
    );

    Args:
        pg_id (Optional[int]): Product group ID, auto-incremented.
        pg_name (str): Name of the product group.
    """

    pg_id: Optional[int] = id_field("productgroups")
    pg_name: str = Field(unique=True)


class Products(SQLModel, table=True):
    """
    CREATE TABLE IF NOT EXISTS products (
        p_id INTEGER PRIMARY KEY DEFAULT nextval('products_id_seq'),
        p_name VARCHAR
    );

    Args:
        p_id (Optional[int]): Product ID, auto-incremented.
        p_name (str): Name of the product.
        pg_id (int): Foreign key referencing the product group.
        perishable (Optional[bool]): Indicates if the product is perishable. Defaults to False
    """

    p_id: Optional[int] = id_field("products")
    p_name: str = Field(unique=True)
    p_group: str = Field(default=None)
    p_cluster: Optional[str] = Field(default=None)
    p_is_perishable: Optional[bool] = Field(default=False)


class Stores(SQLModel, table=True):
    """
    CREATE TABLE IF NOT EXISTS stores (
        s_id INTEGER PRIMARY KEY DEFAULT nextval('stores_id_seq'),
        s_name VARCHAR
    );

    Args:
        s_id (Optional[int]): Store ID, auto-incremented.
        s_name (str): Name of the store.
        s_city (Optional[str]): City where the store is located.
        s_state (Optional[str]): State where the store is located.
        s_country (Optional[str]): Country where the store is located.
        s_type (Optional[str]): Type of the store (e.g., retail, wholesale, online).
        s_cluster (Optional[str]): Cluster of the store (e.g., urban, suburban, rural).
    """

    s_id: Optional[int] = id_field("stores")
    s_name: str = Field(unique=True)
    s_city: Optional[str] = Field(default=None)
    s_state: Optional[str] = Field(default=None)
    s_country: Optional[str] = Field(default=None)
    s_type: Optional[str] = Field(default=None)
    s_cluster: Optional[str] = Field(default=None)
    s_longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    s_latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0)


class Workshops(SQLModel, table=True):
    """
    CREATE TABLE IF NOT EXISTS workshops (
        w_id INTEGER PRIMARY KEY DEFAULT nextval('workshops_id_seq'),
        w_name VARCHAR
    );

    Args:
        w_id (Optional[int]): Workshop ID, auto-incremented.
        w_name (str): Name of the workshop.
    """

    w_id: Optional[int] = id_field("workshops")
    w_name: str = Field(unique=True)
    w_capacity: Optional[int] = Field(default=None, ge=0)
    w_longitude: Optional[float] = Field(default=None, ge=-180.0, le=180.0)
    w_latitude: Optional[float] = Field(default=None, ge=-90.0, le=90.0)


class TransportLinks(SQLModel, table=True):
    """
    CREATE TABLE IF NOT EXISTS transport_links (
        tl_id INTEGER PRIMARY KEY DEFAULT nextval('transport_links_id_seq'),
        tl_w_id INTEGER,
        tl_s_id INTEGER,
        tl_cost DOUBLE DEFAULT 0.0 CHECK (tl_cost >= 0),
        UNIQUE (tl_w_id, tl_s_id),
        FOREIGN KEY (tl_w_id) REFERENCES workshops(w_id),
        FOREIGN KEY (tl_s_id) REFERENCES stores(s_id)
    );

    Args:
        tl_id (Optional[int]): Transport link ID, auto-incremented.
        tl_w_id (int): Foreign key referencing the workshop.
        tl_s_id (int): Foreign key referencing the store.
        tl_p_id (int): Foreign key referencing the product.
        tl_cost (float): Cost of the transport link from workshop to store for the product.
    """

    __table_args__ = (
        UniqueConstraint("tl_w_id", "tl_s_id", "tl_p_id", name="unique_transport_link"),
    )

    tl_id: Optional[int] = id_field("transportlinks")
    tl_p_id: int = Field(foreign_key="products.p_id")
    tl_s_id: int = Field(foreign_key="stores.s_id")
    tl_w_id: int = Field(foreign_key="workshops.w_id")
    tl_package_cost: float = Field(default=None, ge=0.0)
    tl_package_size: int = Field(default=None, ge=0)


class Procurements(SQLModel, table=True):
    """
    CREATE TABLE IF NOT EXISTS procurements (
        pc_id INTEGER PRIMARY KEY DEFAULT nextval('procurements_id_seq'),
        pc_p_id INTEGER,
        pc_s_id INTEGER,
        pc_active_from DATE,
        pc_active_upto DATE CHECK (pc_active_from < pc_active_upto),
        UNIQUE (pc_p_id, pc_s_id, pc_active_from, pc_active_upto),
        FOREIGN KEY (pc_p_id) REFERENCES products(p_id),
        FOREIGN KEY (pc_s_id) REFERENCES stores(s_id)
    );

    Args:
        pc_id (Optional[int]): Procurement ID, auto-incremented.
        pc_p_id (int): Foreign key referencing the product.
        pc_s_id (int): Foreign key referencing the store.
        pc_active_from (date): Start date of the procurement.
        pc_active_upto (date): End date of the procurement.
    """

    __table_args__ = (
        CheckConstraint("pc_active_from < pc_active_upto", name="check_active_dates"),
        UniqueConstraint(
            "pc_p_id",
            "pc_s_id",
            "pc_active_from",
            "pc_active_upto",
            name="unique_procurement",
        ),
    )

    pc_id: Optional[int] = id_field("procurements")
    pc_p_id: int = Field(foreign_key="products.p_id")
    pc_s_id: int = Field(foreign_key="stores.s_id")
    pc_active_from: date
    pc_active_upto: date


class DemandPredictions(SQLModel, table=True):
    """
    CREATE TABLE IF NOT EXISTS demand_predictions (
        dp_id INTEGER PRIMARY KEY DEFAULT nextval('demand_predictions_id_seq'),
        dp_p_id INTEGER,
        dp_s_id INTEGER,
        dp_period DATE,
        dp_mean INTEGER CHECK (dp_mean >= 0),
        UNIQUE (dp_p_id, dp_s_id, dp_period),
        FOREIGN KEY (dp_p_id) REFERENCES products(p_id),
        FOREIGN KEY (dp_s_id) REFERENCES stores(s_id)
    );
    \n
    Args:\n
        dp_id (Optional[int]): Demand prediction ID, auto-incremented.
        dp_p_id (int): Foreign key referencing the product.
        dp_s_id (int): Foreign key referencing the store.
        dp_date (date): Date of the demand prediction.
        dp_mean (int): Mean demand prediction value.
    """

    __table_args__ = (
        UniqueConstraint("dp_p_id", "dp_s_id", "dp_date", name="unique_demand_prediction"),
    )

    dp_id: Optional[int] = id_field("demandpredictions")
    dp_p_id: int = Field(foreign_key="products.p_id")
    dp_s_id: int = Field(foreign_key="stores.s_id")
    dp_date: date
    dp_mean: int = Field(default=None, ge=0.0)


class DemandFulfillments(SQLModel, table=True):
    """
    CREATE TABLE IF NOT EXISTS demand_fulfillments (
        df_id INTEGER PRIMARY KEY DEFAULT nextval('demand_fulfillments_id_seq'),
        df_p_id INTEGER,
        df_s_id INTEGER,
        df_date DATE,
        df_units_sold INTEGER CHECK (df_units_sold >= 0),
        UNIQUE (df_p_id, df_s_id, df_date),
        FOREIGN KEY (df_p_id) REFERENCES products(p_id),
        FOREIGN KEY (df_s_id) REFERENCES stores(s_id)
    );

    Args:
        df_id (Optional[int]): Demand fulfillment ID, auto-incremented.
        df_p_id (int): Foreign key referencing the product.
        df_s_id (int): Foreign key referencing the store.
        df_w_id (int): Foreign key referencing the workshop.
        df_date (date): Date of the demand fulfillment.
        df_units_sent (int): Number of units sent on that date.
    """

    __table_args__ = (
        UniqueConstraint(
            "df_p_id", "df_s_id", "df_w_id", "df_date", name="unique_demand_fulfillment"
        ),
    )

    df_id: Optional[int] = id_field("demandfulfillments")
    df_p_id: int = Field(foreign_key="products.p_id")
    df_s_id: int = Field(foreign_key="stores.s_id")
    df_w_id: int = Field(foreign_key="workshops.w_id")
    df_date: date
    df_packages_sent: int = Field(default=None, ge=0)


class Sales(SQLModel, table=True):
    """
    CREATE TABLE IF NOT EXISTS sales (
        s_id INTEGER PRIMARY KEY DEFAULT nextval('sales_id_seq'),
        s_p_id INTEGER,
        s_s_id INTEGER,
        s_date DATE,
        s_quantity INTEGER CHECK (s_quantity >= 0),
        UNIQUE (s_p_id, s_s_id, s_date),
        FOREIGN KEY (s_p_id) REFERENCES products(p_id),
        FOREIGN KEY (s_s_id) REFERENCES stores(s_id)
    );

    Args:
        s_id (Optional[int]): Sale ID, auto-incremented.
        s_p_id (int): Foreign key referencing the product.
        s_s_id (int): Foreign key referencing the store.
        s_date (date): Date of the sale.
        s_quantity (int): Quantity sold.
    """

    __table_args__ = (UniqueConstraint("sa_p_id", "sa_s_id", "sa_date", name="unique_sale"),)

    sa_id: Optional[int] = id_field("sales")
    sa_p_id: int = Field(foreign_key="products.p_id")
    sa_s_id: int = Field(foreign_key="stores.s_id")
    sa_date: date
    sa_units_sold: int = Field(ge=0)


class Promotions(SQLModel, table=True):
    """
    CREATE TABLE IF NOT EXISTS promotions (
        pr_id INTEGER PRIMARY KEY DEFAULT nextval('promotions_id_seq'),
        pr_p_id INTEGER,
        pr_s_id INTEGER,
        pr_start_date DATE,
        pr_end_date DATE CHECK (pr_start_date < pr_end_date),
        UNIQUE (pr_p_id, pr_s_id, pr_start_date, pr_end_date),
        FOREIGN KEY (pr_p_id) REFERENCES products(p_id),
        FOREIGN KEY (pr_s_id) REFERENCES stores(s_id)
    );

    Args:
        pr_id (Optional[int]): Promotion ID, auto-incremented.
        pr_p_id (int): Foreign key referencing the product.
        pr_s_id (int): Foreign key referencing the store.
        pr_start_date (date): Start date of the promotion.
        pr_end_date (date): End date of the promotion.
    """

    __table_args__ = (
        UniqueConstraint(
            "pr_p_id",
            "pr_s_id",
            "pr_date",
            name="unique_promotion",
        ),
    )

    pr_id: Optional[int] = id_field("promotions")
    pr_p_id: int = Field(foreign_key="products.p_id")
    pr_s_id: int = Field(foreign_key="stores.s_id")
    pr_date: date


class Stocks(SQLModel, table=True):
    """
    CREATE TABLE IF NOT EXISTS stocks (
        sk_id INTEGER PRIMARY KEY DEFAULT nextval('stocks_id_seq'),
        sk_p_id INTEGER,
        sk_s_id INTEGER,
        sk_date DATE,
        sk_starting_inventory INTEGER CHECK (sk_starting_inventory >= 0),
        sk_ending_inventory INTEGER CHECK (sk_ending_inventory >= 0),
        UNIQUE (sk_p_id, sk_s_id, sk_date),
        FOREIGN KEY (sk_p_id) REFERENCES products(p_id),
        FOREIGN KEY (sk_s_id) REFERENCES stores(s_id)
    );

    Args:
        sk_id (Optional[int]): Stock ID, auto-incremented.
        sk_p_id (int): Foreign key referencing the product.
        sk_s_id (int): Foreign key referencing the store.
        sk_date (date): Date of the stock record.
        sk_units (int): Ending inventory for the period in sk_date.
    """

    __table_args__ = (UniqueConstraint("sk_p_id", "sk_s_id", "sk_date", name="unique_stock"),)

    sk_id: Optional[int] = id_field("stocks")
    sk_p_id: int = Field(foreign_key="products.p_id")
    sk_s_id: int = Field(foreign_key="stores.s_id")
    sk_date: date
    sk_units: int = Field(default=0, ge=0)


class Events(SQLModel, table=True):
    """
    CREATE TABLE IF NOT EXISTS events (
        e_id INTEGER PRIMARY KEY DEFAULT nextval('events_id_seq'),
        e_name VARCHAR
    );

    Args:
        e_id (Optional[int]): Event ID, auto-incremented.
        e_name (str): Name of the event.
        e_type (Optional[str]): Type of the event (e.g., holiday, festival). Defaults to None.
    """

    e_id: Optional[int] = id_field("events")
    e_name: str = Field(default=None, unique=True)
    e_type: Optional[str] = Field(default=None)
    e_locality: Optional[str] = Field(default=None)


class EventStores(SQLModel, table=True):
    """
    CREATE TABLE IF NOT EXISTS event_stores (
        es_id INTEGER PRIMARY KEY DEFAULT nextval('event_stores_id_seq'),
        es_e_id INTEGER,
        es_s_id INTEGER,
        UNIQUE (es_e_id, es_s_id),
        FOREIGN KEY (es_e_id) REFERENCES events(e_id),
        FOREIGN KEY (es_s_id) REFERENCES stores(s_id)
    );

    Args:
        es_id (Optional[int]): Event-Store ID, auto-incremented.
        es_e_id (int): Foreign key referencing the event.
        es_s_id (int): Foreign key referencing the store.
    """

    __table_args__ = (UniqueConstraint("es_e_id", "es_s_id", "es_date", name="unique_event_store"),)

    es_id: Optional[int] = id_field("eventstores")
    es_e_id: int = Field(foreign_key="events.e_id")
    es_s_id: int = Field(foreign_key="stores.s_id")
    es_date: date


if __name__ == "__main__":
    from os import path

    from sqlmodel import Session, create_engine

    if not path.exists("data/test.db"):
        engine = create_engine("duckdb:///data/test.db")
        SQLModel.metadata.create_all(engine)
    else:
        engine = create_engine("duckdb:///data/test.db")

    with Session(engine) as session:
        # Upload dummy data

        # 1. Product Groups
        pg1 = ProductGroups(pg_name="BREAD/BAKERY")
        pg2 = ProductGroups(pg_name="DAIRY")
        pgs = [pg1, pg2]
        session.add_all(pgs)
        session.commit()

        # 2. Products
        p1 = Products(p_name="Donut", p_pg_id=pg1.pg_id, perishable=True)
        session.add(p1)

        # 3. Stores
        s1 = Stores(
            s_name="Market A",
            s_city="Quito",
            s_state="Pichincha",
            s_country="Ecuador",
            s_type="D",
            s_cluster="8",
        )
        session.add(s1)
        session.commit()

        # 4. Workshops
        w1 = Workshops(w_name="Bakery A")
        session.add(w1)
        session.commit()

        # 5. Transport Links
        tl1 = TransportLinks(tl_w_id=w1.w_id, tl_s_id=s1.s_id, tl_cost=10.0)
        session.add(tl1)

        # 6. Procurements
        pc1 = Procurements(
            pc_p_id=p1.p_id,
            pc_s_id=s1.s_id,
            pc_active_from=date(2023, 10, 1),
            pc_active_upto=date(2023, 10, 31),
        )
        session.add(pc1)
        session.commit()

        # 7. Demand Predictions
        dp1 = DemandPredictions(
            dp_p_id=p1.p_id,
            dp_s_id=s1.s_id,
            dp_date=date(2023, 10, 15),
            dp_mean=100,
        )
        session.add(dp1)
        session.commit()

        # 8. Stocks
        sk1 = Stocks(
            sk_p_id=p1.p_id,
            sk_s_id=s1.s_id,
            sk_date=date(2023, 10, 15),
            sk_starting_inventory=50,
            sk_ending_inventory=30,
        )
        session.add(sk1)
        session.commit()

        # 9. Events
        e1 = Events(e_name="Bakery Festival")
        session.add(e1)
        session.commit()

        # 10. Event Stores
        es1 = EventStores(es_e_id=e1.e_id, es_s_id=s1.s_id, es_date=date(2023, 10, 20))
        session.add(es1)
        session.commit()

        # print(session.exec(select(ProductGroups)).all())
        # print(session.exec(select(Products)).all())

    engine.dispose()

    import duckdb

    with duckdb.connect(database="./data/test.db", read_only=True) as con:
        pgroups_df = con.execute("""SELECT * FROM productgroups""").pl()
        print(pgroups_df)
        products_df = con.execute("""SELECT * FROM products""").pl()
        print(products_df)
        stores_df = con.execute("""SELECT * FROM stores""").pl()
        print(stores_df)
        workshops_df = con.execute("""SELECT * FROM workshops""").pl()
        print(workshops_df)
        transportlinks_df = con.execute("""SELECT * FROM transportlinks""").pl()
        print(transportlinks_df)
        procurements_df = con.execute("""SELECT * FROM procurements""").pl()
        print(procurements_df)
        demandpredictions_df = con.execute("""SELECT * FROM demandpredictions""").pl()
        print(demandpredictions_df)
        stocks_df = con.execute("""SELECT * FROM stocks""").pl()
        print(stocks_df)
        events_df = con.execute("""SELECT * FROM events""").pl()
        print(events_df)
        eventstores_df = con.execute("""SELECT * FROM eventstores""").pl()
        print(eventstores_df)
