from typing import Any, Optional

import names
import numpy as np
import torch


def filter_by_relation(dataset, relation_id: str):
    return [d for d in dataset if d["requested_rewrite"]["relation_id"] == relation_id]


def get_demonstrations(
    subj_obj_mapping: Optional[list[dict]],
    num_options: int = 3,
    num_icl: int = 5,
    variable_binding_template: str = " {} - {}",  # {variable} - {subject}
    query_template: str = " {} => {}",  # {subject} => {object}.
    used_subjects: Optional[list[str]] = None,
    used_variables: Optional[list[str]] = None,
):
    used_variables = [] if used_variables is None else used_variables
    used_subjects = [] if used_subjects is None else used_subjects
    # print(f"{used_variables=}")
    # print(f"{used_subjects=}")
    demonstrations = []
    answers = []
    subjects_of_interest = []
    variables_of_interest = []
    while len(demonstrations) < num_icl:
        cur_options = [
            (
                subj_obj_mapping[k][0],
                subj_obj_mapping[k][1],
            )
            for k in np.random.choice(
                len(subj_obj_mapping), size=num_options, replace=False
            )
        ]
        # print(used_subjects)
        # print([sub for sub, _ in cur_options])
        used_subjects += [sub for sub, _ in cur_options]
        cur_variables = []
        while len(cur_variables) != num_options:
            name = names.get_first_name()
            if name in used_variables or name in cur_variables:
                continue
            cur_variables.append(name)
        used_variables += cur_variables

        example = (
            ", ".join(
                variable_binding_template.format(name, sub_obj[0])
                for name, sub_obj in zip(cur_variables, cur_options)
            )
            + "."
        )
        query_idx = np.random.choice(num_options)

        if query_template.count("{}") == 2:
            example += query_template.format(
                cur_variables[query_idx], cur_options[query_idx][1]
            )
        elif query_template.count("{}") == 1:
            example += query_template.format(cur_variables[query_idx])
        else:
            raise AssertionError("query_template must contain 1 or 2 `{}`s")
        demonstrations.append(example)

        answers.append(cur_options[query_idx][1])
        subjects_of_interest.append(cur_options[query_idx][0])
        variables_of_interest.append(cur_variables[query_idx])

    used_subjects = list(set(used_subjects))
    used_variables = list(set(used_variables))

    return (
        demonstrations,
        answers,
        subjects_of_interest,
        variables_of_interest,
        used_subjects,
        used_variables,
    )
