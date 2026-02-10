from src.losses.losses import CustomLoss
from src.datasets.simple_dataset import SimpleDataset
from src.ed_models.data_classes import create_data_dict


data_dir = "data/starting_kit_ai-uc_v2/Train_Data"
ed_instance_path = (
    "data/starting_kit_ai-uc_v2/Train_Data/instance_2021_Q1_1/InputData.json"
)
dataset = SimpleDataset(data_dir=data_dir)

data_dict = create_data_dict(ed_instance_path)

loss_fn = CustomLoss(data_dict, violations_penalty=1.0)

for i in range(10):
    data = dataset[2]
    outputs = data["target"]["is_on"].unsqueeze(0)  # add batch dim
    initial_commitment = (
        data["features"]["initial_conditions"].unsqueeze(0)[:, :, -1] > 0
    )
    initial_status = data["features"]["initial_conditions"].unsqueeze(0)[:, :, -1]
    switch_on, switch_off = loss_fn._compute_switch_on_off(
        outputs, initial_commitment=initial_commitment
    )

    supervised_loss = loss_fn.compute_supervised_loss(data["target"], data["target"])

    up_down_violation = loss_fn.compute_constraint_violations(
        outputs, switch_on, switch_off, initial_status=initial_status
    )

    print(f"Supervised Loss: {supervised_loss.item()}")

    print(f"Up/Down Time Violation: {up_down_violation.item()}")
