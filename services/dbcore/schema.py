from datetime import date
from typing import Optional

from sqlalchemy import CheckConstraint, Sequence, UniqueConstraint
from sqlmodel import Field, Session, SQLModel, create_engine


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
        SQLModel (_type_): _description_
        table (bool, optional): _description_. Defaults to True.
    """

    pg_id: Optional[int] = id_field("productgroups")
    pg_name: str


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
    p_name: str
    p_pg_id: int = Field(foreign_key="productgroups.pg_id")
    p_perishable: Optional[bool] = Field(default=False)


class Stores(SQLModel, table=True):
    """
    CREATE TABLE IF NOT EXISTS stores (
        s_id INTEGER PRIMARY KEY DEFAULT nextval('stores_id_seq'),
        s_name VARCHAR
    );

    Args:
        s_id (Optional[int]): Store ID, auto-incremented.
        s_name (str): Name of the store.
    """

    s_id: Optional[int] = id_field("stores")
    s_name: str
    s_city: Optional[str] = Field(default=None)
    s_state: Optional[str] = Field(default=None)
    s_country: Optional[str] = Field(default=None)
    s_type: Optional[str] = Field(default=None)  # e.g., retail, wholesale, online
    s_cluster: Optional[str] = Field(default=None)  # e.g., urban, suburban, rural


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
    w_name: str = Field(default=None)


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
        tl_cost (float): Cost of the transport link.
    """

    __table_args__ = (
        UniqueConstraint("tl_w_id", "tl_s_id", name="unique_transport_link"),
    )

    tl_id: Optional[int] = id_field("transportlinks")
    tl_w_id: int = Field(foreign_key="workshops.w_id")
    tl_s_id: int = Field(foreign_key="stores.s_id")
    tl_cost: float = Field(default=0.0, ge=0.0)


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

    Args:
        dp_id (Optional[int]): Demand prediction ID, auto-incremented.
        dp_p_id (int): Foreign key referencing the product.
        dp_s_id (int): Foreign key referencing the store.
        dp_date (date): Date of the demand prediction.
        dp_mean (int): Mean demand prediction value.
    """

    __table_args__ = (
        UniqueConstraint(
            "dp_p_id", "dp_s_id", "dp_date", name="unique_demand_prediction"
        ),
    )

    dp_id: Optional[int] = id_field("demandpredictions")
    dp_p_id: int = Field(foreign_key="products.p_id")
    dp_s_id: int = Field(foreign_key="stores.s_id")
    dp_date: date
    dp_mean: int = Field(ge=0)


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
        sk_starting_inventory (int): Starting inventory for the period.
        sk_ending_inventory (int): Ending inventory for the period.
    """

    __table_args__ = (
        UniqueConstraint("sk_p_id", "sk_s_id", "sk_date", name="unique_stock"),
    )

    sk_id: Optional[int] = id_field("stocks")
    sk_p_id: int = Field(foreign_key="products.p_id")
    sk_s_id: int = Field(foreign_key="stores.s_id")
    sk_date: date
    sk_starting_inventory: int = Field(ge=0)  # TODO: REMOVE ONE OF THESE
    sk_ending_inventory: int = Field(ge=0)


class Events(SQLModel, table=True):
    """
    CREATE TABLE IF NOT EXISTS events (
        e_id INTEGER PRIMARY KEY DEFAULT nextval('events_id_seq'),
        e_name VARCHAR
    );

    Args:
        e_id (Optional[int]): Event ID, auto-incremented.
        e_name (str): Name of the event.
    """

    e_id: Optional[int] = id_field("events")
    e_name: str


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

    __table_args__ = (
        UniqueConstraint("es_e_id", "es_s_id", "es_date", name="unique_event_store"),
    )

    es_id: Optional[int] = id_field("eventstores")
    es_e_id: int = Field(foreign_key="events.e_id")
    es_s_id: int = Field(foreign_key="stores.s_id")
    es_date: date


if __name__ == "__main__":
    engine = create_engine("duckdb:///data/test.db")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        # Upload dummy data

        # 1. Product Groups
        pg1 = ProductGroups(pg_name="BREAD/BAKERY")
        session.add(pg1)
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
