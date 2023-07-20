// package main

// import (
// 	"fmt"

// 	tf "github.com/galeone/tensorflow/tensorflow/go"
// 	tg "github.com/galeone/tfgo"
// )

// func main() {
// 	fmt.Println("训练流程")
// 	// A model exported with tf.saved_model.save()
// 	// automatically comes with the "serve" tag because the SavedModel
// 	// file format is designed for serving.
// 	// This tag contains the various functions exported. Among these, there is
// 	// always present the "serving_default" signature_def. This signature def
// 	// works exactly like the TF 1.x graph. Get the input tensor and the output tensor,
// 	// and use them as placeholder to feed and output to get, respectively.

// 	// To get info inside a SavedModel the best tool is saved_model_cli
// 	// that comes with the TensorFlow Python package.

// 	// e.g. saved_model_cli show --all --dir output/keras
// 	// gives, among the others, this info:

// 	// signature_def['serving_default']:
// 	// The given SavedModel SignatureDef contains the following input(s):
// 	//   inputs['inputs_input'] tensor_info:
// 	//       dtype: DT_FLOAT
// 	//       shape: (-1, 28, 28, 1)
// 	//       name: serving_default_inputs_input:0
// 	// The given SavedModel SignatureDef contains the following output(s):
// 	//   outputs['logits'] tensor_info:
// 	//       dtype: DT_FLOAT
// 	//       shape: (-1, 10)
// 	//       name: StatefulPartitionedCall:0
// 	// Method name is: tensorflow/serving/predict

// 	model := tg.LoadModel("keras/cnn", []string{"serve"}, nil)

// 	fakeInput, _ := tf.NewTensor([1][28][28][1]float32{})
// 	results := model.Exec([]tf.Output{
// 		model.Op("StatefulPartitionedCall", 0),
// 	}, map[tf.Output]*tf.Tensor{
// 		model.Op("serving_default_inputs_input", 0): fakeInput,
// 	})

// 	predictions := results[0]
// 	fmt.Println(predictions.Value())

// 	fmt.Println("______________________________")

// }
package main

import (
	"fmt"
	"io/ioutil"
	"os"

	tf "github.com/galeone/tensorflow/tensorflow/go"
)

func main() {
	// https://www.imooc.com/wenda/detail/678106
	gm, _ := tf.LoadSavedModel("keras/gm", []string{"serve"}, nil)

	for i, _ := range gm.Graph.Operations() {
		fmt.Println("算子", i, ":", gm.Graph.Operations()[i].Name())
	}

	boolInput, _ := tf.NewTensor([][]float32{{0.5, 0.5, 0.5}, {0, 0, 0}})

	result, _ := gm.Session.Run(
		map[tf.Output]*tf.Tensor{
			gm.Graph.Operation("predict_data").Output(0): boolInput,
		},
		[]tf.Output{
			gm.Graph.Operation("StatefulPartitionedCall_1").Output(0),
		},
		nil,
	)

	floatResults, ok := result[0].Value().([][]float32)
	if !ok {
		fmt.Println("No float results")
	}

	fmt.Println(floatResults)

	trainData, _ := tf.NewTensor([][]float32{
		{0.5, 0.5, 0.5},
		{0.5, 0.5, 0.5},
		{0.5, 0.5, 0.5},
		{0.5, 0.5, 0.5},
		{0.5, 0.5, 0.5},
		{0, 0, 0},
		{0, 0, 0},
		{0, 0, 0},
		{0, 0, 0},
		{0, 0, 0},
	})
	trainLabels, _ := tf.NewTensor([][]float32{
		{0},
		{0},
		{0},
		{0},
	})

	for i := 0; i < 1000; i++ {
		_, _ = gm.Session.Run(
			map[tf.Output]*tf.Tensor{
				gm.Graph.Operation("learn_data").Output(0):   trainData,
				gm.Graph.Operation("learn_labels").Output(0): trainLabels,
			},
			[]tf.Output{
				gm.Graph.Operation("StatefulPartitionedCall").Output(0),
			},
			nil,
		)
	}

	boolTest, _ := tf.NewTensor([][]float32{{0.5, 0.5, 0.5}, {0, 0, 0}})

	test, _ := gm.Session.Run(
		map[tf.Output]*tf.Tensor{
			gm.Graph.Operation("predict_data").Output(0): boolTest,
		},
		[]tf.Output{
			gm.Graph.Operation("StatefulPartitionedCall_1").Output(0),
		},
		nil,
	)

	testResults, ok := test[0].Value().([][]float32)
	if !ok {
		print(testResults)
		fmt.Println("No post training float results")
	}

	fmt.Println(testResults)

	os.RemoveAll("keras/gm-trained")
	os.MkdirAll("keras/gm-trained/variables", os.ModePerm)
	savedModel, _ := ioutil.ReadFile("keras/gm/saved_model.pb")

	ioutil.WriteFile("keras/gm-trained/saved_model.pb", savedModel, os.ModePerm)

	filenameInput, _ := tf.NewTensor("keras/gm-trained/variables/variables")

	_, _ = gm.Session.Run(
		map[tf.Output]*tf.Tensor{
			gm.Graph.Operation("saver_filename").Output(0): filenameInput,
		},
		[]tf.Output{
			gm.Graph.Operation("StatefulPartitionedCall_2").Output(0),
		},
		nil,
	)

	gmTrained, _ := tf.LoadSavedModel("keras/gm-trained", []string{"serve"}, nil)

	boolTest, _ = tf.NewTensor([][]float32{{0.5, 0.5, 0.5}, {0, 0, 0}})

	test, _ = gmTrained.Session.Run(
		map[tf.Output]*tf.Tensor{
			gmTrained.Graph.Operation("predict_data").Output(0): boolTest,
		},
		[]tf.Output{
			gmTrained.Graph.Operation("StatefulPartitionedCall_1").Output(0),
		},
		nil,
	)

	testResults, ok = test[0].Value().([][]float32)
	fmt.Println(testResults)
	if !ok {
		fmt.Println("No post training float results")
	}

	fmt.Println(testResults)

	fmt.Println("Done")
}
