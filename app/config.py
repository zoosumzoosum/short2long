from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    youtube_api_key: str = ""  # worker만 씀 (api 컨테이너에는 없어도 됨)


settings = Settings()
