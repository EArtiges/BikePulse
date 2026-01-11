import oslo_lib

city = "Edinburgh"
providers = {
    "Oslo": "oslobysykkel.no",
    "Edinburgh": "edinburghcyclehire.com",
    "Milan": "bikemi.com",
}
provider = providers[city]

df = oslo_lib.collect_data(
    years=[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025], months=range(1, 13), provider=provider
)
df.to_pickle(f"data/{city}/trips/{city.lower()}_data.pkl")
