import json


def main():
    with open("datapackage.json") as f:
        data = json.load(f)

        dump = open("details.txt", "w")
        for table in data["resources"]:
            table_name = table["path"].replace("data/", "").replace(".csv", "")

            if "description" in table:
                dump.write(
                    "comment on table "
                    + table_name
                    + " is '"
                    + table["description"]
                    + "'\n"
                )

            for column in table["schema"]["fields"]:
                if "description" in column:
                    description = (
                        column["description"].replace("\n", ". ").replace("..", ".")
                    )
                    dump.write(
                        "comment on column "
                        + table_name
                        + "."
                        + column["id"]
                        + " is '"
                        + description
                        + "'\n"
                    )

            dump.write("\n\n")

        dump.close()


if __name__ == "__main__":
    main()
