from pathlib import Path


def root_directory(path: str = '.') -> str:
    """
    Returns the projects root-directory, determined by the existence of the `.git` directory.

    :param path: The path to start the search. Default is current directory.
    :return:
    """
    file = Path(f"{path}/.git")

    if file.is_dir():
        return path
    else:
        return root_directory(f"{path}/..")


def strip_margin(s: str) -> str:
    """
    Equivalent to Scala's stripMargin String method.

    :param s: A multiline string. (Every line should begin with '|')
    :return:
    """
    return str.join("\n", [line.strip()[2:] for line in s.split("\n")]).strip()
