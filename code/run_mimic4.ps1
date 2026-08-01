param(
    [ValidateSet("knn", "cluster")]
    [string]$Task = "knn",

    [ValidateSet("train", "test")]
    [string]$Action = "train",

    [string]$DataDir = "..\data\mimic4",
    [string]$CacheDir = "preprocess_data\mimic4",
    [string]$OutputDir = "",
    [string]$ResumePath = "",
    [string]$PythonBin = "python"
)
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "1"
$env:PYTHONUTF8 = "1"
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = "res\mimic4\$Task"
}

if ($Task -eq "cluster") {
    $TrainCache = "$CacheDir\train_cluster.pt"
}
else {
    $TrainCache = "$CacheDir\train_knn.pt"
}

$ActionArgs = @()
if ($Action -eq "train") {
    $ActionArgs += "--do_train"
}
else {
    if ([string]::IsNullOrWhiteSpace($ResumePath)) {
        throw "Set -ResumePath to the trained checkpoint when -Action test is used."
    }
    $ActionArgs += @("--do_test", "--resume_path", $ResumePath)
}

$TrainArgs = @(
    "train.py",
    "--dataset", "mimic4",
    "--data_dir", $DataDir,
    "--task", $Task,
    "--use_conv", "anti",
    "--gcn_conv_nums", "2",
    "--hidden_size", "100",
    "--pair_neurons", "30",
    "--hidden_dropout_prob", "0.4",
    "--batch_size", "256",
    "--epoch", "30",
    "--learning_rate", "0.0001",
    "--knn_k_values", "1", "3", "5",
    "--selection_k", "1",
    "--train_data", $TrainCache,
    "--train_input_data", "$CacheDir\train_input_knn.pt",
    "--valid_cluster_data", "$CacheDir\valid_cluster.pt",
    "--valid_knn_data", "$CacheDir\valid_knn.pt",
    "--test_cluster_data", "$CacheDir\test_cluster.pt",
    "--test_knn_data", "$CacheDir\test_knn.pt",
    "--output_dir", $OutputDir
) + $ActionArgs

Push-Location $PSScriptRoot
try {
    Write-Host "Running MHGRL: task=$Task action=$Action"
    & $PythonBin @TrainArgs
    if ($LASTEXITCODE -ne 0) {
        throw "MHGRL exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
