import sys
import pandas as pd
from lightgbm import Booster
from catboost import CatBoostClassifier


def days_between_invites(days: list):
    if len(days) == 1:
        return -100, -100, -100, -100, -100
    diff_days = []
    for i in range(1, len(days)):
        diff_days.append((days[i] - days[i-1]).days)
    if len(diff_days) == 1:
        return diff_days[0], -100, sum(diff_days) / len(days), diff_days[-1], -100
    return diff_days[0], diff_days[1], sum(diff_days) / len(days), diff_days[-1], diff_days[-2]


def clients_invited_last_n_days(days, end_date):
    return lambda x: sum((end_date - d).days <= days for d in x)


def make_dataset(input_df: pd.DataFrame, end_date):
    input_df = input_df.copy()
    output_df = input_df.groupby("clientbankpartner_pin").agg({"client_pin": "count", "partnerrolestart_date": "min", "client_start_date": ["min", "max", "median", lambda x: sorted(list(x))]})

    # Кол-во привлечений в неделю/месяц
    output_df["average_month_invites"] = output_df["client_pin"]["count"]/ output_df["client_start_date"]["<lambda_0>"].agg(lambda x: len(set(map(lambda y: y.month, x))))
    output_df["average_week_invites"] = output_df["client_pin"]["count"]/ output_df["client_start_date"]["<lambda_0>"].agg(lambda x: len(set(map(lambda y: y.week, x))))
    # Сколько дней проходит между первым-вторым, предпоследним-последним привлечением
    output_df[["diff_day_first", "diff_day_second", "diff_days_mean", "diff_days_last", "diff_days_prelast"]] = output_df["client_start_date"]["<lambda_0>"].apply(days_between_invites).tolist()
    # Сколько дней прошло с последнего/медианного привлечения
    output_df["diff_test_date_and_last_invite"] = end_date - output_df["client_start_date"]["max"]
    output_df["end_date_clientstart_median"] = end_date - output_df["client_start_date"]["median"]
    # Сколько дней партнер всего привлекает людей
    output_df["end_date_clientstart_max-min"] = output_df["client_start_date"]["max"] - output_df["client_start_date"]["min"]
    output_df["difference_median_and_max-min"] = output_df["end_date_clientstart_max-min"] - output_df["end_date_clientstart_median"]

    # Перевод дат в целые числа
    output_df["diff_first_lastdt"] = output_df["diff_day_first"] - output_df["diff_days_last"]
    output_df["diff_between_end_date_and_first_client"] = end_date - output_df["client_start_date"]["min"]
    output_df["diff_test_date_and_last_invite"] = output_df["diff_test_date_and_last_invite"].dt.days
    output_df["end_date_clientstart_median"] = output_df["end_date_clientstart_median"].dt.days
    output_df["end_date_clientstart_max-min"] = output_df["end_date_clientstart_max-min"].dt.days
    output_df["difference_median_and_max-min"] = output_df["difference_median_and_max-min"].dt.days
    output_df["diff_between_end_date_and_first_client"] = output_df["diff_between_end_date_and_first_client"].dt.days

    # Сколько людей привлекли за последние 30/60/90/180/270/365 дней
    output_df["clients_per_day"] = output_df["diff_between_end_date_and_first_client"] / output_df["client_pin"]["count"]
    output_df["clients_invited_last_30_days"] = output_df["client_start_date"]["<lambda_0>"].apply(clients_invited_last_n_days(30, end_date))
    output_df["clients_invited_last_60_days"] = output_df["client_start_date"]["<lambda_0>"].apply(clients_invited_last_n_days(60, end_date))
    output_df["clients_invited_last_90_days"] = output_df["client_start_date"]["<lambda_0>"].apply(clients_invited_last_n_days(90, end_date))
    output_df["clients_invited_last_180_days"] = output_df["client_start_date"]["<lambda_0>"].apply(clients_invited_last_n_days(180, end_date))
    output_df["clients_invited_last_270_days"] = output_df["client_start_date"]["<lambda_0>"].apply(clients_invited_last_n_days(270, end_date))
    output_df["clients_invited_last_365_days"] = output_df["client_start_date"]["<lambda_0>"].apply(clients_invited_last_n_days(365, end_date))

    output_df = output_df.drop(columns=[("client_start_date", "<lambda_0>"), ("partnerrolestart_date", "min"), ("client_start_date", "min"), ("client_start_date", "max"), ("client_start_date", "median"), ('client_pin', 'count')])

    return output_df


def get_args():
    input_file_path = input("Please pass the path to the test dataset.\nFor example: '/dataset.csv'\n")
    test_date = input("Please pass the reporting date.\nFor example: 2020-12-01\n")
    output_file_path = input("Please pass the path to the test dataset.\nFor example: '/result.csv'\n")

    return input_file_path, test_date, output_file_path


def main(input_file, test_date, output_file):
    df = pd.read_csv(input_file)
    df["client_start_date"] = pd.to_datetime(df["client_start_date"])
    df["partnerrolestart_date"] = pd.to_datetime(df["partnerrolestart_date"])
    test_dataset = make_dataset(df, pd.to_datetime(test_date))
    lgbm_model = Booster(model_file="lgbm.txt")
    catboost_model = CatBoostClassifier()
    catboost_model.load_model("catboost.cbm")
    lgbm_prediction = 1 - lgbm_model.predict(test_dataset)
    catboost_prediction = catboost_model.predict_proba(test_dataset)[:, 0]
    preds = pd.DataFrame({"clientbankpartner_pin": test_dataset.index, "score": lgbm_prediction})
    preds["score"] = lgbm_prediction * 0.3 + catboost_prediction * 0.70
    preds.to_csv(output_file, index=False)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        input_file, test_date, output_file = get_args()
    elif len(sys.argv) > 4:
        print("You need pass only 3 argument: path to the test dataset, reporting date and output file."
              "\nFor example: 'python3 make_prediction.py /dataset.csv 2020-12-01 /result.csv")
        exit()
    else:
        input_file, test_date, output_file = sys.argv[1:]

    main(input_file, test_date, output_file)
    print("The prediction completed")
