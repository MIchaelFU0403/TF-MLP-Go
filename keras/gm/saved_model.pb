э 
л¤
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.0.02v2.0.0-rc2-26-g64c3d382ca8лж
d
VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Variable
]
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
shape: *
dtype0	*
_output_shapes
: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
shared_nameAdam/beta_1*
shape: *
dtype0
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
dtype0*
shape: *
shared_nameAdam/beta_2*
_output_shapes
: 
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
shared_name
Adam/decay*
shape: *
dtype0
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
dtype0*
_output_shapes
: 
t
dense/kernelVarHandleOp*
dtype0*
shape
:*
_output_shapes
: *
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes

:
l

dense/biasVarHandleOp*
_output_shapes
: *
shape:*
shared_name
dense/bias*
dtype0
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
В
Adam/dense/kernel/mVarHandleOp*$
shared_nameAdam/dense/kernel/m*
_output_shapes
: *
dtype0*
shape
:
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

:*
dtype0
z
Adam/dense/bias/mVarHandleOp*"
shared_nameAdam/dense/bias/m*
_output_shapes
: *
dtype0*
shape:
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
dtype0*
_output_shapes
:
В
Adam/dense/kernel/vVarHandleOp*
dtype0*
_output_shapes
: *
shape
:*$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
dtype0*
_output_shapes

:
z
Adam/dense/bias/vVarHandleOp*
shape:*"
shared_nameAdam/dense/bias/v*
dtype0*
_output_shapes
: 
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
dtype0*
_output_shapes
:

NoOpNoOp
░
ConstConst"/device:CPU:0*
_output_shapes
: *ы
valueсB▐ B╫
=
	model
_global_step

_optimizer

signatures
Ж
layer-0
layer_with_weights-0
layer-1
regularization_losses
	variables
	trainable_variables

	keras_api
EC
VARIABLE_VALUEVariable'_global_step/.ATTRIBUTES/VARIABLE_VALUE
d
iter

beta_1

beta_2
	decay
learning_ratem&m'v(v)
 
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
 

0
1

0
1
Ъ

layers
layer_regularization_losses
regularization_losses
non_trainable_variables
	variables
	trainable_variables
metrics
IG
VARIABLE_VALUE	Adam/iter*_optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEAdam/beta_1,_optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEAdam/beta_2,_optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUE
Adam/decay+_optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEAdam/learning_rate3_optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
Ъ

layers
layer_regularization_losses
regularization_losses
 non_trainable_variables
	variables
trainable_variables
!metrics
^\
VARIABLE_VALUEdense/kernel<model/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUE
dense/bias:model/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
Ъ

"layers
#layer_regularization_losses
regularization_losses
$non_trainable_variables
	variables
trainable_variables
%metrics

0
1
 
 
 
 
 
 
 
 
 
 
 
ГА
VARIABLE_VALUEAdam/dense/kernel/mYmodel/layer_with_weights-0/kernel/.OPTIMIZER_SLOT/_optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense/bias/mWmodel/layer_with_weights-0/bias/.OPTIMIZER_SLOT/_optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ГА
VARIABLE_VALUEAdam/dense/kernel/vYmodel/layer_with_weights-0/kernel/.OPTIMIZER_SLOT/_optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense/bias/vWmodel/layer_with_weights-0/bias/.OPTIMIZER_SLOT/_optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0
m

learn_dataPlaceholder*
shape:         *'
_output_shapes
:         *
dtype0
o
learn_labelsPlaceholder*
dtype0*'
_output_shapes
:         *
shape:         
╦
StatefulPartitionedCallStatefulPartitionedCall
learn_datalearn_labelsVariabledense/kernel
dense/biasAdam/learning_rate	Adam/iterAdam/beta_1Adam/beta_2Adam/dense/kernel/mAdam/dense/kernel/vAdam/dense/bias/mAdam/dense/bias/v**
config_proto

CPU

GPU 2J 8*
Tin
2**
_gradient_op_typePartitionedCall-625**
f%R#
!__inference_signature_wrapper_445*#
_output_shapes
:         *
Tout
2
o
predict_dataPlaceholder*
dtype0*'
_output_shapes
:         *
shape:         
и
StatefulPartitionedCall_1StatefulPartitionedCallpredict_datadense/kernel
dense/bias*'
_output_shapes
:         **
_gradient_op_typePartitionedCall-628**
config_proto

CPU

GPU 2J 8*
Tout
2*
Tin
2**
f%R#
!__inference_signature_wrapper_456
O
saver_filenamePlaceholder*
shape: *
_output_shapes
: *
dtype0
╡
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameVariable/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst**
config_proto

CPU

GPU 2J 8**
_gradient_op_typePartitionedCall-662*
Tout
2*
_output_shapes
: *
Tin
2	*%
f R
__inference__traced_save_661
└
StatefulPartitionedCall_3StatefulPartitionedCallsaver_filenameVariable	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratedense/kernel
dense/biasAdam/dense/kernel/mAdam/dense/bias/mAdam/dense/kernel/vAdam/dense/bias/v*
Tin
2**
_gradient_op_typePartitionedCall-711*(
f#R!
__inference__traced_restore_710*
_output_shapes
: **
config_proto

CPU

GPU 2J 8*
Tout
2└ь
Ў
╫
>__inference_dense_layer_call_and_return_conditional_losses_485

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         *
T0Й
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:         *
T0"
identityIdentity:output:0*.
_input_shapes
:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
а
╤
>__inference_model_layer_call_and_return_conditional_losses_539

inputs(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identityИвdense/StatefulPartitionedCallї
dense/StatefulPartitionedCallStatefulPartitionedCallinputs$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tout
2*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_485*'
_output_shapes
:         *
Tin
2**
config_proto

CPU

GPU 2J 8**
_gradient_op_typePartitionedCall-491О
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*.
_input_shapes
:         ::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
а
╤
>__inference_model_layer_call_and_return_conditional_losses_522

inputs(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identityИвdense/StatefulPartitionedCallї
dense/StatefulPartitionedCallStatefulPartitionedCallinputs$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tout
2*'
_output_shapes
:         *
Tin
2**
config_proto

CPU

GPU 2J 8*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_485**
_gradient_op_typePartitionedCall-491О
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*.
_input_shapes
:         ::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
я	
я
>__inference_model_layer_call_and_return_conditional_losses_555

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpо
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0u
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         м
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ы
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*'
_output_shapes
:         *
T0"
identityIdentity:output:0*.
_input_shapes
:         ::2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
╒
и
#__inference_model_layer_call_fn_545

bool_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCall
bool_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_539*'
_output_shapes
:         **
_gradient_op_typePartitionedCall-540**
config_proto

CPU

GPU 2J 8В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall: :* &
$
_user_specified_name
bool_input: 
╒
и
#__inference_model_layer_call_fn_528

bool_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCall
bool_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_522*
Tin
2*'
_output_shapes
:         **
_gradient_op_typePartitionedCall-523**
config_proto

CPU

GPU 2J 8*
Tout
2В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall: :* &
$
_user_specified_name
bool_input: 
╘

ы
__inference__wrapped_model_469

bool_input.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource
identityИв"model/dense/BiasAdd/ReadVariableOpв!model/dense/MatMul/ReadVariableOp║
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:Е
model/dense/MatMulMatMul
bool_input)model/dense/MatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0╕
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0Ъ
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         *
T0н
IdentityIdentitymodel/dense/BiasAdd:output:0#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0*.
_input_shapes
:         ::2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp: :* &
$
_user_specified_name
bool_input: 
я	
я
>__inference_model_layer_call_and_return_conditional_losses_565

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpо
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:u
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         м
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0И
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ы
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0*.
_input_shapes
:         ::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
м
╒
>__inference_model_layer_call_and_return_conditional_losses_503

bool_input(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identityИвdense/StatefulPartitionedCall∙
dense/StatefulPartitionedCallStatefulPartitionedCall
bool_input$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*'
_output_shapes
:         *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_485**
config_proto

CPU

GPU 2J 8*
Tin
2*
Tout
2**
_gradient_op_typePartitionedCall-491О
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*.
_input_shapes
:         ::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall: :* &
$
_user_specified_name
bool_input: 
╤Й
О
__inference_learn_410
data

labels 
assignaddvariableop_resource.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource%
!adam_cast_readvariableop_resource 
adam_readvariableop_resource'
#adam_cast_2_readvariableop_resource'
#adam_cast_3_readvariableop_resource(
$adam_adam_update_resourceapplyadam_m(
$adam_adam_update_resourceapplyadam_v*
&adam_adam_update_1_resourceapplyadam_m*
&adam_adam_update_1_resourceapplyadam_v
identityИвAdam/Adam/AssignAddVariableOpв$Adam/Adam/update/Read/ReadVariableOpв"Adam/Adam/update/ResourceApplyAdamв&Adam/Adam/update_1/Read/ReadVariableOpв$Adam/Adam/update_1/ResourceApplyAdamвAdam/Cast/ReadVariableOpвAdam/Cast_2/ReadVariableOpвAdam/Cast_3/ReadVariableOpвAdam/ReadVariableOpвAssignAddVariableOpв"model/dense/BiasAdd/ReadVariableOpв!model/dense/MatMul/ReadVariableOpG
ConstConst*
dtype0*
_output_shapes
: *
value	B :{
AssignAddVariableOpAssignAddVariableOpassignaddvariableop_resourceConst:output:0*
_output_shapes
 *
dtype0║
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0
model/dense/MatMulMatMuldata)model/dense/MatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0╕
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Ъ
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         L
Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *    L
Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *Х┐╓3J
sub/xConst*
_output_shapes
: *
valueB
 *  А?*
dtype0M
subSubsub/x:output:0Const_2:output:0*
T0*
_output_shapes
: y
clip_by_value/MinimumMinimummodel/dense/BiasAdd:output:0sub:z:0*'
_output_shapes
:         *
T0w
clip_by_valueMaximumclip_by_value/Minimum:z:0Const_2:output:0*
T0*'
_output_shapes
:         J
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3a
addAddV2clip_by_value:z:0add/y:output:0*
T0*'
_output_shapes
:         E
LogLogadd:z:0*'
_output_shapes
:         *
T0M
mulMullabelsLog:y:0*'
_output_shapes
:         *
T0L
sub_1/xConst*
_output_shapes
: *
valueB
 *  А?*
dtype0X
sub_1Subsub_1/x:output:0labels*'
_output_shapes
:         *
T0L
sub_2/xConst*
_output_shapes
: *
valueB
 *  А?*
dtype0c
sub_2Subsub_2/x:output:0clip_by_value:z:0*'
_output_shapes
:         *
T0L
add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *Х┐╓3]
add_1AddV2	sub_2:z:0add_1/y:output:0*
T0*'
_output_shapes
:         I
Log_1Log	add_1:z:0*'
_output_shapes
:         *
T0T
mul_1Mul	sub_1:z:0	Log_1:y:0*
T0*'
_output_shapes
:         T
add_2AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         G
NegNeg	add_2:z:0*'
_output_shapes
:         *
T0a
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
         d
MeanMeanNeg:y:0Mean/reduction_indices:output:0*
T0*#
_output_shapes
:         B
ShapeShapeMean:output:0*
T0*
_output_shapes
:O

ones/ConstConst*
valueB
 *  А?*
dtype0*
_output_shapes
: _
onesFillShape:output:0ones/Const:output:0*#
_output_shapes
:         *
T0>
Shape_1ShapeNeg:y:0*
_output_shapes
:*
T0b
SizeConst*
_output_shapes
: *
dtype0*
value	B :*
_class
loc:@Shape_1{
add_3AddV2Mean/reduction_indices:output:0Size:output:0*
_output_shapes
: *
_class
loc:@Shape_1*
T0f
modFloorMod	add_3:z:0Size:output:0*
_class
loc:@Shape_1*
_output_shapes
: *
T0f
Shape_2Const*
_output_shapes
: *
_class
loc:@Shape_1*
valueB *
dtype0i
range/startConst*
dtype0*
value	B : *
_class
loc:@Shape_1*
_output_shapes
: i
range/deltaConst*
_class
loc:@Shape_1*
_output_shapes
: *
value	B :*
dtype0Б
rangeRangerange/start:output:0Size:output:0range/delta:output:0*
_output_shapes
:*
_class
loc:@Shape_1h

Fill/valueConst*
_class
loc:@Shape_1*
_output_shapes
: *
dtype0*
value	B :p
FillFillShape_2:output:0Fill/value:output:0*
_output_shapes
: *
_class
loc:@Shape_1*
T0в
DynamicStitchDynamicStitchrange:output:0mod:z:0Shape_1:output:0Fill:output:0*
_output_shapes
:*
N*
_class
loc:@Shape_1*
T0g
	Maximum/yConst*
value	B :*
dtype0*
_class
loc:@Shape_1*
_output_shapes
: 
MaximumMaximumDynamicStitch:merged:0Maximum/y:output:0*
_output_shapes
:*
T0*
_class
loc:@Shape_1t
floordivFloorDivShape_1:output:0Maximum:z:0*
_output_shapes
:*
_class
loc:@Shape_1*
T0t
ReshapeReshapeones:output:0DynamicStitch:merged:0*
T0*0
_output_shapes
:                  g
TileTileReshape:output:0floordiv:z:0*
T0*0
_output_shapes
:                  >
Shape_3ShapeNeg:y:0*
T0*
_output_shapes
:D
Shape_4ShapeMean:output:0*
_output_shapes
:*
T0Q
Const_3Const*
_output_shapes
:*
valueB: *
dtype0Q
ProdProdShape_3:output:0Const_3:output:0*
_output_shapes
: *
T0Q
Const_4Const*
dtype0*
_output_shapes
:*
valueB: S
Prod_1ProdShape_4:output:0Const_4:output:0*
T0*
_output_shapes
: M
Maximum_1/yConst*
_output_shapes
: *
dtype0*
value	B :\
	Maximum_1MaximumProd_1:output:0Maximum_1/y:output:0*
_output_shapes
: *
T0U

floordiv_1FloorDivProd:output:0Maximum_1:z:0*
T0*
_output_shapes
: L
CastCastfloordiv_1:z:0*

SrcT0*
_output_shapes
: *

DstT0f
truedivRealDivTile:output:0Cast:y:0*
T0*0
_output_shapes
:                  T
Neg_1Negtruediv:z:0*
T0*0
_output_shapes
:                  >
Shape_5Shapemul:z:0*
_output_shapes
:*
T0@
Shape_6Shape	mul_1:z:0*
T0*
_output_shapes
:Ж
BroadcastGradientArgsBroadcastGradientArgsShape_5:output:0Shape_6:output:0*2
_output_shapes 
:         :         T
SumSum	Neg_1:y:0BroadcastGradientArgs:r0:0*
T0*
_output_shapes
:f
	Reshape_1ReshapeSum:output:0Shape_5:output:0*'
_output_shapes
:         *
T0V
Sum_1Sum	Neg_1:y:0BroadcastGradientArgs:r1:0*
T0*
_output_shapes
:h
	Reshape_2ReshapeSum_1:output:0Shape_6:output:0*'
_output_shapes
:         *
T0=
Shape_7Shapelabels*
_output_shapes
:*
T0>
Shape_8ShapeLog:y:0*
T0*
_output_shapes
:И
BroadcastGradientArgs_1BroadcastGradientArgsShape_7:output:0Shape_8:output:0*2
_output_shapes 
:         :         Z
Mul_2MullabelsReshape_1:output:0*'
_output_shapes
:         *
T0X
Sum_2Sum	Mul_2:z:0BroadcastGradientArgs_1:r1:0*
_output_shapes
:*
T0h
	Reshape_3ReshapeSum_2:output:0Shape_8:output:0*
T0*'
_output_shapes
:         @
Shape_9Shape	sub_1:z:0*
T0*
_output_shapes
:A
Shape_10Shape	Log_1:y:0*
_output_shapes
:*
T0Й
BroadcastGradientArgs_2BroadcastGradientArgsShape_9:output:0Shape_10:output:0*2
_output_shapes 
:         :         ]
Mul_3Mul	sub_1:z:0Reshape_2:output:0*
T0*'
_output_shapes
:         X
Sum_3Sum	Mul_3:z:0BroadcastGradientArgs_2:r1:0*
T0*
_output_shapes
:i
	Reshape_4ReshapeSum_3:output:0Shape_10:output:0*'
_output_shapes
:         *
T0_

Reciprocal
Reciprocaladd:z:0
^Reshape_3*
T0*'
_output_shapes
:         b
mul_4MulReshape_3:output:0Reciprocal:y:0*'
_output_shapes
:         *
T0c
Reciprocal_1
Reciprocal	add_1:z:0
^Reshape_4*'
_output_shapes
:         *
T0d
mul_5MulReshape_4:output:0Reciprocal_1:y:0*'
_output_shapes
:         *
T0F
Shape_11Shapesub_2/x:output:0*
T0*
_output_shapes
: I
Shape_12Shapeclip_by_value:z:0*
_output_shapes
:*
T0К
BroadcastGradientArgs_3BroadcastGradientArgsShape_11:output:0Shape_12:output:0*2
_output_shapes 
:         :         I
Neg_2Neg	mul_5:z:0*'
_output_shapes
:         *
T0X
Sum_4Sum	Neg_2:y:0BroadcastGradientArgs_3:r1:0*
_output_shapes
:*
T0i
	Reshape_5ReshapeSum_4:output:0Shape_12:output:0*'
_output_shapes
:         *
T0f
AddNAddN	mul_4:z:0Reshape_5:output:0*
T0*
N*'
_output_shapes
:         U

zeros_like	ZerosLike
AddN:sum:0*'
_output_shapes
:         *
T0{
GreaterEqualGreaterEqualclip_by_value/Minimum:z:0Const_2:output:0*'
_output_shapes
:         *
T0p
SelectSelectGreaterEqual:z:0
AddN:sum:0zeros_like:y:0*'
_output_shapes
:         *
T0\
zeros_like_1	ZerosLikeSelect:output:0*
T0*'
_output_shapes
:         o
	LessEqual	LessEqualmodel/dense/BiasAdd:output:0sub:z:0*'
_output_shapes
:         *
T0v
Select_1SelectLessEqual:z:0Select:output:0zeros_like_1:y:0*
T0*'
_output_shapes
:         R
BiasAddGradBiasAddGradSelect_1:output:0*
T0*
_output_shapes
:e
MatMulMatMuldataSelect_1:output:0*
transpose_a(*
_output_shapes

:*
T0а
Adam/Cast/ReadVariableOpReadVariableOp!adam_cast_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: К
Adam/IdentityIdentity Adam/Cast/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
T0Ц
Adam/ReadVariableOpReadVariableOpadam_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0	z

Adam/add/yConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0	*
value	B	 RТ
Adam/addAddV2Adam/ReadVariableOp:value:0Adam/add/y:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0	*
_output_shapes
: 
Adam/Cast_1CastAdam/add:z:0",/job:localhost/replica:0/task:0/device:CPU:0*

SrcT0	*
_output_shapes
: *

DstT0д
Adam/Cast_2/ReadVariableOpReadVariableOp#adam_cast_2_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0О
Adam/Identity_1Identity"Adam/Cast_2/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: д
Adam/Cast_3/ReadVariableOpReadVariableOp#adam_cast_3_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: О
Adam/Identity_2Identity"Adam/Cast_3/ReadVariableOp:value:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
T0Й
Adam/PowPowAdam/Identity_1:output:0Adam/Cast_1:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: Л

Adam/Pow_1PowAdam/Identity_2:output:0Adam/Cast_1:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: }

Adam/sub/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *  А?Г
Adam/subSubAdam/sub/x:output:0Adam/Pow_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
T0n
	Adam/SqrtSqrtAdam/sub:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
T0
Adam/sub_1/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
valueB
 *  А?*
_output_shapes
: *
dtype0Е

Adam/sub_1SubAdam/sub_1/x:output:0Adam/Pow:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
T0Е
Adam/truedivRealDivAdam/Sqrt:y:0Adam/sub_1:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
T0И
Adam/mulMulAdam/Identity:output:0Adam/truediv:z:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: }

Adam/ConstConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
valueB
 *Х┐╓3*
dtype0
Adam/sub_2/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
valueB
 *  А?*
dtype0С

Adam/sub_2SubAdam/sub_2/x:output:0Adam/Identity_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
: *
T0
Adam/sub_3/xConst",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
: *
valueB
 *  А?С

Adam/sub_3SubAdam/sub_3/x:output:0Adam/Identity_2:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*
_output_shapes
: с
$Adam/Adam/update/Read/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource"^model/dense/MatMul/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0|
Adam/Adam/update/IdentityIdentity,Adam/Adam/update/Read/ReadVariableOp:value:0*
_output_shapes

:*
T0А
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam*model_dense_matmul_readvariableop_resource$adam_adam_update_resourceapplyadam_m$adam_adam_update_resourceapplyadam_vAdam/Pow:z:0Adam/Pow_1:z:0Adam/Identity:output:0Adam/Identity_1:output:0Adam/Identity_2:output:0Adam/Const:output:0MatMul:product:0%^Adam/Adam/update/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@Adam/Adam/update/Read/ReadVariableOp*
use_locking(*
_output_shapes
 с
&Adam/Adam/update_1/Read/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource#^model/dense/BiasAdd/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:|
Adam/Adam/update_1/IdentityIdentity.Adam/Adam/update_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:П
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam+model_dense_biasadd_readvariableop_resource&adam_adam_update_1_resourceapplyadam_m&adam_adam_update_1_resourceapplyadam_vAdam/Pow:z:0Adam/Pow_1:z:0Adam/Identity:output:0Adam/Identity_1:output:0Adam/Identity_2:output:0Adam/Const:output:0BiasAddGrad:output:0'^Adam/Adam/update_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 *
use_locking(*
T0*9
_class/
-+loc:@Adam/Adam/update_1/Read/ReadVariableOpЭ
Adam/Adam/ConstConst#^Adam/Adam/update/ResourceApplyAdam%^Adam/Adam/update_1/ResourceApplyAdam*
_output_shapes
: *
dtype0	*
value	B	 Rе
Adam/Adam/AssignAddVariableOpAssignAddVariableOpadam_readvariableop_resourceAdam/Adam/Const:output:0^Adam/ReadVariableOp*
_output_shapes
 *
dtype0	╫
IdentityIdentityMean:output:0^Adam/Adam/AssignAddVariableOp%^Adam/Adam/update/Read/ReadVariableOp#^Adam/Adam/update/ResourceApplyAdam'^Adam/Adam/update_1/Read/ReadVariableOp%^Adam/Adam/update_1/ResourceApplyAdam^Adam/Cast/ReadVariableOp^Adam/Cast_2/ReadVariableOp^Adam/Cast_3/ReadVariableOp^Adam/ReadVariableOp^AssignAddVariableOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp*#
_output_shapes
:         *
T0"
identityIdentity:output:0*e
_input_shapesT
R:         :         :::::::::::2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2*
Adam/ReadVariableOpAdam/ReadVariableOp2*
AssignAddVariableOpAssignAddVariableOp2P
&Adam/Adam/update_1/Read/ReadVariableOp&Adam/Adam/update_1/Read/ReadVariableOp2>
Adam/Adam/AssignAddVariableOpAdam/Adam/AssignAddVariableOp24
Adam/Cast/ReadVariableOpAdam/Cast/ReadVariableOp28
Adam/Cast_2/ReadVariableOpAdam/Cast_2/ReadVariableOp2H
"Adam/Adam/update/ResourceApplyAdam"Adam/Adam/update/ResourceApplyAdam2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$Adam/Adam/update/Read/ReadVariableOp$Adam/Adam/update/Read/ReadVariableOp28
Adam/Cast_3/ReadVariableOpAdam/Cast_3/ReadVariableOp2L
$Adam/Adam/update_1/ResourceApplyAdam$Adam/Adam/update_1/ResourceApplyAdam: : :&"
 
_user_specified_namelabels: :=	9
7
_class-
+)loc:@Adam/Adam/update/Read/ReadVariableOp:?;
9
_class/
-+loc:@Adam/Adam/update_1/Read/ReadVariableOp: : :$  

_user_specified_namedata: :?;
9
_class/
-+loc:@Adam/Adam/update_1/Read/ReadVariableOp: :=
9
7
_class-
+)loc:@Adam/Adam/update/Read/ReadVariableOp
Э3
й
__inference__traced_restore_710
file_prefix
assignvariableop_variable 
assignvariableop_1_adam_iter"
assignvariableop_2_adam_beta_1"
assignvariableop_3_adam_beta_2!
assignvariableop_4_adam_decay)
%assignvariableop_5_adam_learning_rate#
assignvariableop_6_dense_kernel!
assignvariableop_7_dense_bias*
&assignvariableop_8_adam_dense_kernel_m(
$assignvariableop_9_adam_dense_bias_m+
'assignvariableop_10_adam_dense_kernel_v)
%assignvariableop_11_adam_dense_bias_v
identity_13ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9в	RestoreV2вRestoreV2_1ф
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*К
valueАB¤B'_global_step/.ATTRIBUTES/VARIABLE_VALUEB*_optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB,_optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB,_optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB+_optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB3_optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB<model/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB:model/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEBYmodel/layer_with_weights-0/kernel/.OPTIMIZER_SLOT/_optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWmodel/layer_with_weights-0/bias/.OPTIMIZER_SLOT/_optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBYmodel/layer_with_weights-0/kernel/.OPTIMIZER_SLOT/_optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWmodel/layer_with_weights-0/bias/.OPTIMIZER_SLOT/_optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0И
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B ┌
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
dtypes
2	*D
_output_shapes2
0::::::::::::L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:u
AssignVariableOpAssignVariableOpassignvariableop_variableIdentity:output:0*
_output_shapes
 *
dtype0N

Identity_1IdentityRestoreV2:tensors:1*
_output_shapes
:*
T0	|
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_iterIdentity_1:output:0*
_output_shapes
 *
dtype0	N

Identity_2IdentityRestoreV2:tensors:2*
_output_shapes
:*
T0~
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_1Identity_2:output:0*
_output_shapes
 *
dtype0N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:~
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_2Identity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:}
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_decayIdentity_4:output:0*
_output_shapes
 *
dtype0N

Identity_5IdentityRestoreV2:tensors:5*
_output_shapes
:*
T0Е
AssignVariableOp_5AssignVariableOp%assignvariableop_5_adam_learning_rateIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
_output_shapes
:*
T0
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0*
_output_shapes
 *
dtype0N

Identity_7IdentityRestoreV2:tensors:7*
_output_shapes
:*
T0}
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0*
_output_shapes
 *
dtype0N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:Ж
AssignVariableOp_8AssignVariableOp&assignvariableop_8_adam_dense_kernel_mIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:Д
AssignVariableOp_9AssignVariableOp$assignvariableop_9_adam_dense_bias_mIdentity_9:output:0*
_output_shapes
 *
dtype0P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:Й
AssignVariableOp_10AssignVariableOp'assignvariableop_10_adam_dense_kernel_vIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:З
AssignVariableOp_11AssignVariableOp%assignvariableop_11_adam_dense_bias_vIdentity_11:output:0*
_output_shapes
 *
dtype0М
RestoreV2_1/tensor_namesConst"/device:CPU:0*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B ╡
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
dtypes
2*
_output_shapes
:1
NoOpNoOp"/device:CPU:0*
_output_shapes
 ╫
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ф
Identity_13IdentityIdentity_12:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_13Identity_13:output:0*E
_input_shapes4
2: ::::::::::::2(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112
RestoreV2_1RestoreV2_12(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_6AssignVariableOp_6: : : : :	 : : : :+ '
%
_user_specified_namefile_prefix: : : :
 
╔
д
#__inference_model_layer_call_fn_579

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*'
_output_shapes
:         **
_gradient_op_typePartitionedCall-540*
Tout
2*G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_539*
Tin
2В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
╔
д
#__inference_model_layer_call_fn_572

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8*
Tin
2*'
_output_shapes
:         **
_gradient_op_typePartitionedCall-523*
Tout
2*G
fBR@
>__inference_model_layer_call_and_return_conditional_losses_522В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
┌#
м
__inference__traced_save_661
file_prefix'
#savev2_variable_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_1_const

identity_1ИвMergeV2CheckpointsвSaveV2вSaveV2_1О
StringJoin/inputs_1Const"/device:CPU:0*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_b219ad0956224f7497359b1d2c6ba68c/parts

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
dtype0*
value	B : *
_output_shapes
: У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: с
SaveV2/tensor_namesConst"/device:CPU:0*К
valueАB¤B'_global_step/.ATTRIBUTES/VARIABLE_VALUEB*_optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB,_optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB,_optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB+_optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB3_optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB<model/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB:model/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEBYmodel/layer_with_weights-0/kernel/.OPTIMIZER_SLOT/_optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBWmodel/layer_with_weights-0/bias/.OPTIMIZER_SLOT/_optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBYmodel/layer_with_weights-0/kernel/.OPTIMIZER_SLOT/_optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBWmodel/layer_with_weights-0/bias/.OPTIMIZER_SLOT/_optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:Е
SaveV2/shape_and_slicesConst"/device:CPU:0*+
value"B B B B B B B B B B B B B *
dtype0*
_output_shapes
:к
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0#savev2_variable_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
2	h
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
value	B :*
dtype0Ч
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Й
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
_output_shapes
:*
dtype0q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:├
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2╣
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
_output_shapes
:*
T0*
NЦ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*S
_input_shapesB
@: : : : : : : ::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1: : : : : :	 : : : :+ '
%
_user_specified_namefile_prefix: : : :
 
Ў
╫
>__inference_dense_layer_call_and_return_conditional_losses_589

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         *
T0Й
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:         *
T0"
identityIdentity:output:0*.
_input_shapes
:         ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
║

▐
__inference_predict_425
data.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource
identityИв"model/dense/BiasAdd/ReadVariableOpв!model/dense/MatMul/ReadVariableOp║
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:
model/dense/MatMulMatMuldata)model/dense/MatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0╕
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:*
dtype0Ъ
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         н
IdentityIdentitymodel/dense/BiasAdd:output:0#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp*'
_output_shapes
:         *
T0"
identityIdentity:output:0*.
_input_shapes
:         ::2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp: :$  

_user_specified_namedata: 
Ъ
а
!__inference_signature_wrapper_456
data"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCall║
StatefulPartitionedCallStatefulPartitionedCalldatastatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

CPU

GPU 2J 8* 
fR
__inference_predict_425*'
_output_shapes
:         *
Tin
2*
Tout
2**
_gradient_op_typePartitionedCall-451В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall: :$  

_user_specified_namedata: 
м
╒
>__inference_model_layer_call_and_return_conditional_losses_512

bool_input(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identityИвdense/StatefulPartitionedCall∙
dense/StatefulPartitionedCallStatefulPartitionedCall
bool_input$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*
Tin
2*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*
Tout
2*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_485**
_gradient_op_typePartitionedCall-491О
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*.
_input_shapes
:         ::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall: :* &
$
_user_specified_name
bool_input: 
╔
д
#__inference_dense_layer_call_fn_596

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*'
_output_shapes
:         **
config_proto

CPU

GPU 2J 8*
Tin
2*
Tout
2*G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_485**
_gradient_op_typePartitionedCall-491В
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
м
є
!__inference_signature_wrapper_445
data

labels"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12
identityИвStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCalldatalabelsstatefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12*
Tin
2**
_gradient_op_typePartitionedCall-431*
Tout
2*
fR
__inference_learn_410**
config_proto

CPU

GPU 2J 8*#
_output_shapes
:         ~
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:         "
identityIdentity:output:0*e
_input_shapesT
R:         :         :::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : :&"
 
_user_specified_namelabels: :	 : : : :$  

_user_specified_namedata: : : :
 "wL
saver_filename:0StatefulPartitionedCall_2:0StatefulPartitionedCall_38"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*║
learn░
/
labels%
learn_labels:0         
+
data#
learn_data:0         4
loss,
StatefulPartitionedCall:0         tensorflow/serving/predict*Щ
predictН
-
data%
predict_data:0         @

prediction2
StatefulPartitionedCall_1:0         tensorflow/serving/predict:ЕQ
s
	model
_global_step

_optimizer

signatures
	*learn
+predict"
_generic_user_object
ї
layer-0
layer_with_weights-0
layer-1
regularization_losses
	variables
	trainable_variables

	keras_api
,_default_save_signature
*-&call_and_return_all_conditional_losses
.__call__"Х
_tf_keras_model√{"class_name": "Model", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [10, 3], "dtype": "float32", "sparse": false, "name": "bool_input"}, "name": "bool_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["bool_input", 0, 0, {}]]]}], "input_layers": [["bool_input", 0, 0]], "output_layers": [["dense", 0, 0]]}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [10, 3], "dtype": "float32", "sparse": false, "name": "bool_input"}, "name": "bool_input", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["bool_input", 0, 0, {}]]]}], "input_layers": [["bool_input", 0, 0]], "output_layers": [["dense", 0, 0]]}}}
: 2Variable
w
iter

beta_1

beta_2
	decay
learning_ratem&m'v(v)"
	optimizer
/
	/learn
0predict"
signature_map
г
regularization_losses
	variables
trainable_variables
	keras_api
*1&call_and_return_all_conditional_losses
2__call__"Ф
_tf_keras_layer·{"class_name": "InputLayer", "name": "bool_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [10, 3], "config": {"batch_input_shape": [10, 3], "dtype": "float32", "sparse": false, "name": "bool_input"}}
э

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*3&call_and_return_all_conditional_losses
4__call__"╚
_tf_keras_layerо{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3}}}}
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
╖

layers
layer_regularization_losses
regularization_losses
non_trainable_variables
	variables
	trainable_variables
metrics
.__call__
,_default_save_signature
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ

layers
layer_regularization_losses
regularization_losses
 non_trainable_variables
	variables
trainable_variables
!metrics
2__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
:2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Ъ

"layers
#layer_regularization_losses
regularization_losses
$non_trainable_variables
	variables
trainable_variables
%metrics
4__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
#:!2Adam/dense/kernel/m
:2Adam/dense/bias/m
#:!2Adam/dense/kernel/v
:2Adam/dense/bias/v
№2∙
__inference_learn_410▀
б▓Э
FullArgSpec%
argsЪ
jself
jdata
jlabels
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *7в4
К         
К         
┌2╫
__inference_predict_425╗
Ч▓У
FullArgSpec
argsЪ
jself
jdata
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в
К         
▀2▄
__inference__wrapped_model_469╣
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *)в&
$К!

bool_input         
╞2├
>__inference_model_layer_call_and_return_conditional_losses_565
>__inference_model_layer_call_and_return_conditional_losses_512
>__inference_model_layer_call_and_return_conditional_losses_503
>__inference_model_layer_call_and_return_conditional_losses_555└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
┌2╫
#__inference_model_layer_call_fn_528
#__inference_model_layer_call_fn_579
#__inference_model_layer_call_fn_572
#__inference_model_layer_call_fn_545└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
3B1
!__inference_signature_wrapper_445datalabels
-B+
!__inference_signature_wrapper_456data
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
╠2╔╞
╜▓╣
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkwjkwargs
defaultsЪ 

kwonlyargsЪ

jtraining%
kwonlydefaultsк

trainingp 
annotationsк *
 
ш2х
>__inference_dense_layer_call_and_return_conditional_losses_589в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
═2╩
#__inference_dense_layer_call_fn_596в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 Ю
>__inference_dense_layer_call_and_return_conditional_losses_589\/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ v
#__inference_dense_layer_call_fn_596O/в,
%в"
 К
inputs         
к "К         З
__inference_predict_425l-в*
#в 
К
data         
к "7к4
2

prediction$К!

prediction         ~
#__inference_model_layer_call_fn_572W7в4
-в*
 К
inputs         
p

 
к "К         ┐
!__inference_signature_wrapper_445Щ&(')aв^
в 
WкT
*
labels К
labels         
&
dataК
data         "'к$
"
lossК
loss         ж
>__inference_model_layer_call_and_return_conditional_losses_565d7в4
-в*
 К
inputs         
p 

 
к "%в"
К
0         
Ъ В
#__inference_model_layer_call_fn_528[;в8
1в.
$К!

bool_input         
p

 
к "К         б
__inference_learn_410З&(')OвL
EвB
К
data         
 К
labels         
к "'к$
"
lossК
loss         ~
#__inference_model_layer_call_fn_579W7в4
-в*
 К
inputs         
p 

 
к "К         к
>__inference_model_layer_call_and_return_conditional_losses_503h;в8
1в.
$К!

bool_input         
p

 
к "%в"
К
0         
Ъ ж
>__inference_model_layer_call_and_return_conditional_losses_555d7в4
-в*
 К
inputs         
p

 
к "%в"
К
0         
Ъ В
#__inference_model_layer_call_fn_545[;в8
1в.
$К!

bool_input         
p 

 
к "К         К
__inference__wrapped_model_469h3в0
)в&
$К!

bool_input         
к "-к*
(
denseК
dense         Щ
!__inference_signature_wrapper_456t5в2
в 
+к(
&
dataК
data         "7к4
2

prediction$К!

prediction         к
>__inference_model_layer_call_and_return_conditional_losses_512h;в8
1в.
$К!

bool_input         
p 

 
к "%в"
К
0         
Ъ 