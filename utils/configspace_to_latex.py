import pandas as pd
import ConfigSpace as CS


pd.options.display.max_colwidth = 100
type_map = dict(
    UniformFloatHyperparameter="float",
    UniformIntegerHyperparameter="int",
    CategoricalHyperparameter="cat",
    OrdinalHyperparameter="ord"
)


def configspace_to_latex(configspace):
    columns = ["name", "type", "log", "range"]
    dfs = []
    for hp in configspace.get_hyperparameters():
        vals = [hp.name, type_map[type(hp).__name__]]
        if hasattr(hp, "log"):
            vals.append(hp.log)
        else:
            vals.append("NA")
        if isinstance(hp, (CS.UniformFloatHyperparameter, CS.UniformIntegerHyperparameter)):
            lower, upper = hp.lower, hp.upper
            choices = "-"
            if isinstance(hp, CS.UniformIntegerHyperparameter):
                lower = int(lower)
                upper = int(upper)
            range = [lower, upper]
        else:
            lower, upper = "-", "-"
            if isinstance(hp, CS.CategoricalHyperparameter):
                range = hp.choices
            else:
                range = hp.sequence
        range = [str(i) for i in range]
        vals.append("[{}]".format(", ".join(range)))
        dfs.append(pd.DataFrame([vals], columns=columns))
    df = pd.concat(dfs)
    print(df.to_latex(index=False))

