package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"

	tf "github.com/galeone/tensorflow/tensorflow/go"
)

const (
	// Initial file for MLP model.
	MLPInitialFilePath = "d7y/models/MLP"

	// Ratio of training dataset for MLP model.
	TrainRatio = 0.8

	// Batch size for each optimization step for MLP model.
	BatchSize int = 32

	// EpochCount for MLP model.
	EpochCount int = 10

	// The dimension of score Tensor for MLP model's inputs.
	ScoreDims int = 5
)

func main() {
	a, _ := os.Getwd()
	fmt.Println(a)

	// 相对路径
	model, err := tf.LoadSavedModel("d7y/models/MLP", []string{"serve"}, nil)
	if err != nil {
		fmt.Println("fail to load model")
		return
	}

	for i, _ := range model.Graph.Operations() {
		fmt.Println("OP_ID", i, ":", model.Graph.Operations()[i].Name())
	}

	// MLPObservationFile, err := os.Open(filepath.Join(a, "MLP_training.csv"))
	MLPObservationFile, err := os.Open(filepath.Join(a, "d7y/MLP_training.csv"))

	if err != nil {
		// return err
		fmt.Println("read file error")
		return
	}
	defer MLPObservationFile.Close()
	// MLPTestingFile, err := os.Open(filepath.Join(a, "MLP_testing.csv"))
	MLPTestingFile, err := os.Open(filepath.Join(a, "d7y/MLP_testing.csv"))
	if err != nil {
		// return err
		fmt.Println("read file error")
		return
	}
	defer MLPTestingFile.Close()

	trainReader := csv.NewReader(MLPObservationFile)
	testReader := csv.NewReader(MLPTestingFile)

	// Initialize score and target batch.
	scoresBatch := make([][]float64, 0, BatchSize)
	targetBatch := make([]float64, 0, BatchSize)

	fmt.Println("start training")
	for i := 0; i < EpochCount; i++ {
		fmt.Println("epoch count", i)
		for {
			// Read preprocessed file in line.
			line, err := trainReader.Read()
			if err == io.EOF {
				MLPObservationFile.Seek(0, 0)
				break
			}

			if err != nil {
				return
				// return err
			}

			scores := make([]float64, len(line)-1)
			// Add scores and target of each MLP observation to training batch.
			for j, v := range line[:len(line)-1] {
				var val float64
				if val, err = strconv.ParseFloat(v, 64); err != nil {
					return
					// return err
				}
				scores[j] = val
			}
			scoresBatch = append(scoresBatch, scores)

			target, err := strconv.ParseFloat(line[len(line)-1], 64)
			if err != nil {
				// return err
				return
			}
			targetBatch = append(targetBatch, target)

			if len(scoresBatch) == BatchSize && len(targetBatch) == BatchSize {
				// Convert MLP observation to Tensorflow tensor.
				scoreTensor, targetTensor, err := ObservationToTensor(scoresBatch, targetBatch, BatchSize)
				if err != nil {
					// return err
					return
				}

				// Run an optimization step on the model using batch of values from observation file.
				if err := learnMLP(model, scoreTensor, targetTensor); err != nil {
					// return err
					return
				}

				// Reset score and target batch for further training phase.
				scoresBatch = make([][]float64, 0, BatchSize)
				targetBatch = make([]float64, 0, BatchSize)
			}
		}
	}
	fmt.Println("training over")

	var accMSE float64
	var accMAE float64
	var testCount int

	fmt.Println("start testing")

	for {
		// Read preprocessed file in line.
		line, err := testReader.Read()

		if err == io.EOF {
			break
		}

		if err != nil {
			return
		}

		scores := make([][]float64, 1)
		scores[0] = make([]float64, len(line)-1)
		target := make([]float64, 1)
		// Add scores and target of each MLP observation to training batch.
		for j, v := range line[:len(line)-1] {
			val, err := strconv.ParseFloat(v, 64)
			if err != nil {
				return
				// return err
			}
			scores[0][j] = val
		}

		target[0], err = strconv.ParseFloat(line[len(line)-1], 64)
		if err != nil {
			return
		}

		// Convert MLP observation to Tensorflow tensor.
		scoreTensor, targetTensor, err := ObservationToTensor(scores, target, 1)
		if err != nil {
			return
		}

		// Run an optimization step on the model using batch of values from observation file.
		MSE, MAE, err := testMLP(model, scoreTensor, targetTensor)
		if err != nil {
			return
		}
		accMSE += MSE[0]
		accMAE += MAE[0]
		testCount++
	}

	MAE := accMAE / float64(testCount)
	MSE := accMSE / float64(testCount)
	fmt.Println("Metric-MAE:", MAE)
	fmt.Println("Metric-MSE:", MSE)

	// Save the trained model.
	os.RemoveAll(filepath.Join(a, "d7y/Models/MLP_trained"))
	os.MkdirAll(filepath.Join(a, "d7y/Models/MLP_trained/variables"), os.ModePerm)

	// Saved model file unchanged.
	originSavedModelFile, err := os.OpenFile(filepath.Join(a, "d7y/Models/MLP/saved_model.pb"), os.O_RDWR|os.O_CREATE|os.O_APPEND, 0600)
	if err != nil {
		fmt.Println("origin saved model file open error")

		return
	}

	savedModelFile, err := os.OpenFile(filepath.Join(a, "d7y/Models/MLP_trained/saved_model.pb"), os.O_RDWR|os.O_CREATE|os.O_APPEND, 0600)
	if err != nil {
		fmt.Println("saved model file open error")

		return
	}

	if _, err := io.Copy(savedModelFile, originSavedModelFile); err != nil {
		fmt.Println("write save model file error")

		return
	}

	// Update variables file
	variableTensor, err := tf.NewTensor("d7y/Models/MLP_trained/variables/variables")
	if err != nil {
		fmt.Println("load variables error")

		return
	}

	if err := saveMLP(model, variableTensor); err != nil {
		fmt.Println("save variable file error")

		return
	}

}

// testMLP tests the model using batch of values from observation file.
func testMLP(model *tf.SavedModel, scoreTensor, targetTensor *tf.Tensor) ([]float64, []float64, error) {
	res, err := model.Session.Run(
		map[tf.Output]*tf.Tensor{
			model.Graph.Operation("MAE_inputs").Output(0):  scoreTensor,
			model.Graph.Operation("MAE_targets").Output(0): targetTensor,
		},
		[]tf.Output{
			model.Graph.Operation("StatefulPartitionedCall").Output(0),
		},
		nil,
	)

	if err != nil {
		fmt.Println("test and get MLP model MAE metric error: %s", err.Error())

		return nil, nil, err
	}
	mae := res[0].Value().([]float64)

	res, err = model.Session.Run(
		map[tf.Output]*tf.Tensor{
			model.Graph.Operation("MSE_inputs").Output(0):  scoreTensor,
			model.Graph.Operation("MSE_targets").Output(0): targetTensor,
		},
		[]tf.Output{
			model.Graph.Operation("StatefulPartitionedCall_1").Output(0),
		},
		nil,
	)

	if err != nil {
		fmt.Println("test and get MLP model MSE metric error: %s", err.Error())

		return nil, nil, err
	}
	mse := res[0].Value().([]float64)

	return mse, mae, nil
}

// learnMLP runs an optimization step on the model using batch of values from observation file.
func learnMLP(model *tf.SavedModel, scoreTensor, targetTensor *tf.Tensor) error {
	res, err := model.Session.Run(
		map[tf.Output]*tf.Tensor{
			model.Graph.Operation("train_inputs").Output(0):  scoreTensor,
			model.Graph.Operation("train_targets").Output(0): targetTensor,
		},
		[]tf.Output{
			model.Graph.Operation("StatefulPartitionedCall_3").Output(0),
			model.Graph.Operation("StatefulPartitionedCall_3").Output(1),
		},
		nil,
	)

	loss := res[0].Value().([]float64)
	fmt.Println("loss:", loss)
	if err != nil {
		// logger.Error("learn MLP model error: %s", err.Error())
		fmt.Println("learn MLP model error: %s", err.Error())

		return err
	}

	return nil
}

// saveMLP saves trained MLP model weight into files.
func saveMLP(model *tf.SavedModel, variables *tf.Tensor) error {
	_, err := model.Session.Run(
		map[tf.Output]*tf.Tensor{
			model.Graph.Operation("saver_filename").Output(0): variables,
		},
		[]tf.Output{
			model.Graph.Operation("StatefulPartitionedCall_4").Output(0),
		},
		nil,
	)

	if err != nil {
		fmt.Println("save MLP model error: %s", err.Error())

		return err
	}

	return nil
}

// ObservationToTensor converts MLP train observations to Tensorflow tensors.
func ObservationToTensor(scoresBatch [][]float64, targetBatch []float64, batchSize int) (*tf.Tensor, *tf.Tensor, error) {
	scores := make([][]float64, batchSize)
	target := make([]float64, batchSize)

	for i := 0; i < batchSize; i++ {
		scores[i] = make([]float64, ScoreDims)
		copy(scores[i], scoresBatch[i])
		target[i] = targetBatch[i]
	}

	scoreTensor, err := tf.NewTensor(scores)
	if err != nil {
		// logger.Errorf("create score tensor failed: %v", err)
		fmt.Println("learn score tensor failed: %s", err.Error())

		return nil, nil, err
	}

	targetTensor, err := tf.NewTensor(target)
	if err != nil {
		fmt.Println("create target tensor failed: %s", err.Error())

		return nil, nil, err
	}

	return scoreTensor, targetTensor, nil
}
