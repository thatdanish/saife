
g
is_training/Initializer/ConstConst*
_class
loc:@is_training*
value	B
 Z *
dtype0

w
is_training
VariableV2*
shape: *
shared_name *
_class
loc:@is_training*
dtype0
*
	container 

is_training/AssignAssignis_trainingis_training/Initializer/Const*
use_locking(*
T0
*
_class
loc:@is_training*
validate_shape(
R
is_training/readIdentityis_training*
T0
*
_class
loc:@is_training
6
Assign/valueConst*
value	B
 Z*
dtype0

}
AssignAssignis_trainingAssign/value*
use_locking(*
T0
*
_class
loc:@is_training*
validate_shape(
8
Assign_1/valueConst*
value	B
 Z *
dtype0


Assign_1Assignis_trainingAssign_1/value*
use_locking(*
T0
*
_class
loc:@is_training*
validate_shape(
H
	input_imgPlaceholder*!
shape:’’’’’’’’’
*
dtype0
I

target_imgPlaceholder*!
shape:’’’’’’’’’
*
dtype0
I
input_img_labelPlaceholder*
shape:’’’’’’’’’*
dtype0
4
	keep_probPlaceholder*
shape:*
dtype0
I
latent_variablePlaceholder*
shape:’’’’’’’’’2*
dtype0
F
prior_samplePlaceholder*
shape:’’’’’’’’’2*
dtype0
L
prior_sample_labelPlaceholder*
shape:’’’’’’’’’*
dtype0
Z
CNN_encoder_cat/Reshape/shapeConst*%
valueB"’’’’
        *
dtype0
c
CNN_encoder_cat/ReshapeReshape	input_imgCNN_encoder_cat/Reshape/shape*
T0*
Tshape0
£
9CNN_encoder_cat/Conv2D/W/Initializer/random_uniform/shapeConst*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*%
valueB"             *
dtype0

7CNN_encoder_cat/Conv2D/W/Initializer/random_uniform/minConst*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
valueB
 *:Ķæ*
dtype0

7CNN_encoder_cat/Conv2D/W/Initializer/random_uniform/maxConst*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
valueB
 *:Ķ?*
dtype0
é
ACNN_encoder_cat/Conv2D/W/Initializer/random_uniform/RandomUniformRandomUniform9CNN_encoder_cat/Conv2D/W/Initializer/random_uniform/shape*

seed *
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
dtype0*
seed2 
ę
7CNN_encoder_cat/Conv2D/W/Initializer/random_uniform/subSub7CNN_encoder_cat/Conv2D/W/Initializer/random_uniform/max7CNN_encoder_cat/Conv2D/W/Initializer/random_uniform/min*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W
š
7CNN_encoder_cat/Conv2D/W/Initializer/random_uniform/mulMulACNN_encoder_cat/Conv2D/W/Initializer/random_uniform/RandomUniform7CNN_encoder_cat/Conv2D/W/Initializer/random_uniform/sub*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W
ā
3CNN_encoder_cat/Conv2D/W/Initializer/random_uniformAdd7CNN_encoder_cat/Conv2D/W/Initializer/random_uniform/mul7CNN_encoder_cat/Conv2D/W/Initializer/random_uniform/min*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W
”
CNN_encoder_cat/Conv2D/W
VariableV2*
shape: *
shared_name *+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
dtype0*
	container 
×
CNN_encoder_cat/Conv2D/W/AssignAssignCNN_encoder_cat/Conv2D/W3CNN_encoder_cat/Conv2D/W/Initializer/random_uniform*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(
y
CNN_encoder_cat/Conv2D/W/readIdentityCNN_encoder_cat/Conv2D/W*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W

*CNN_encoder_cat/Conv2D/b/Initializer/ConstConst*+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
valueB *    *
dtype0

CNN_encoder_cat/Conv2D/b
VariableV2*
shape: *
shared_name *+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
dtype0*
	container 
Ī
CNN_encoder_cat/Conv2D/b/AssignAssignCNN_encoder_cat/Conv2D/b*CNN_encoder_cat/Conv2D/b/Initializer/Const*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
validate_shape(
y
CNN_encoder_cat/Conv2D/b/readIdentityCNN_encoder_cat/Conv2D/b*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b
ļ
CNN_encoder_cat/Conv2D/Conv2DConv2DCNN_encoder_cat/ReshapeCNN_encoder_cat/Conv2D/W/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME

CNN_encoder_cat/Conv2D/BiasAddBiasAddCNN_encoder_cat/Conv2D/Conv2DCNN_encoder_cat/Conv2D/b/read*
T0*
data_formatNHWC
L
CNN_encoder_cat/Conv2D/TanhTanhCNN_encoder_cat/Conv2D/BiasAdd*
T0
¤
!CNN_encoder_cat/MaxPool2D/MaxPoolMaxPoolCNN_encoder_cat/Conv2D/Tanh*
ksize
*
paddingSAME*
T0*
data_formatNHWC*
strides

§
;CNN_encoder_cat/Conv2D_1/W/Initializer/random_uniform/shapeConst*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*%
valueB"              *
dtype0

9CNN_encoder_cat/Conv2D_1/W/Initializer/random_uniform/minConst*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
valueB
 *ģŃ½*
dtype0

9CNN_encoder_cat/Conv2D_1/W/Initializer/random_uniform/maxConst*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
valueB
 *ģŃ=*
dtype0
ļ
CCNN_encoder_cat/Conv2D_1/W/Initializer/random_uniform/RandomUniformRandomUniform;CNN_encoder_cat/Conv2D_1/W/Initializer/random_uniform/shape*

seed *
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0*
seed2 
ī
9CNN_encoder_cat/Conv2D_1/W/Initializer/random_uniform/subSub9CNN_encoder_cat/Conv2D_1/W/Initializer/random_uniform/max9CNN_encoder_cat/Conv2D_1/W/Initializer/random_uniform/min*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W
ų
9CNN_encoder_cat/Conv2D_1/W/Initializer/random_uniform/mulMulCCNN_encoder_cat/Conv2D_1/W/Initializer/random_uniform/RandomUniform9CNN_encoder_cat/Conv2D_1/W/Initializer/random_uniform/sub*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W
ź
5CNN_encoder_cat/Conv2D_1/W/Initializer/random_uniformAdd9CNN_encoder_cat/Conv2D_1/W/Initializer/random_uniform/mul9CNN_encoder_cat/Conv2D_1/W/Initializer/random_uniform/min*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W
„
CNN_encoder_cat/Conv2D_1/W
VariableV2*
shape:  *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0*
	container 
ß
!CNN_encoder_cat/Conv2D_1/W/AssignAssignCNN_encoder_cat/Conv2D_1/W5CNN_encoder_cat/Conv2D_1/W/Initializer/random_uniform*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
validate_shape(

CNN_encoder_cat/Conv2D_1/W/readIdentityCNN_encoder_cat/Conv2D_1/W*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W

,CNN_encoder_cat/Conv2D_1/b/Initializer/ConstConst*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
valueB *    *
dtype0

CNN_encoder_cat/Conv2D_1/b
VariableV2*
shape: *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
dtype0*
	container 
Ö
!CNN_encoder_cat/Conv2D_1/b/AssignAssignCNN_encoder_cat/Conv2D_1/b,CNN_encoder_cat/Conv2D_1/b/Initializer/Const*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
validate_shape(

CNN_encoder_cat/Conv2D_1/b/readIdentityCNN_encoder_cat/Conv2D_1/b*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b
ż
CNN_encoder_cat/Conv2D_1/Conv2DConv2D!CNN_encoder_cat/MaxPool2D/MaxPoolCNN_encoder_cat/Conv2D_1/W/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME

 CNN_encoder_cat/Conv2D_1/BiasAddBiasAddCNN_encoder_cat/Conv2D_1/Conv2DCNN_encoder_cat/Conv2D_1/b/read*
T0*
data_formatNHWC
P
CNN_encoder_cat/Conv2D_1/TanhTanh CNN_encoder_cat/Conv2D_1/BiasAdd*
T0
Ø
#CNN_encoder_cat/MaxPool2D_1/MaxPoolMaxPoolCNN_encoder_cat/Conv2D_1/Tanh*
ksize
*
paddingSAME*
T0*
data_formatNHWC*
strides

§
;CNN_encoder_cat/Conv2D_2/W/Initializer/random_uniform/shapeConst*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*%
valueB"              *
dtype0

9CNN_encoder_cat/Conv2D_2/W/Initializer/random_uniform/minConst*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
valueB
 *ģŃ½*
dtype0

9CNN_encoder_cat/Conv2D_2/W/Initializer/random_uniform/maxConst*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
valueB
 *ģŃ=*
dtype0
ļ
CCNN_encoder_cat/Conv2D_2/W/Initializer/random_uniform/RandomUniformRandomUniform;CNN_encoder_cat/Conv2D_2/W/Initializer/random_uniform/shape*

seed *
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0*
seed2 
ī
9CNN_encoder_cat/Conv2D_2/W/Initializer/random_uniform/subSub9CNN_encoder_cat/Conv2D_2/W/Initializer/random_uniform/max9CNN_encoder_cat/Conv2D_2/W/Initializer/random_uniform/min*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W
ų
9CNN_encoder_cat/Conv2D_2/W/Initializer/random_uniform/mulMulCCNN_encoder_cat/Conv2D_2/W/Initializer/random_uniform/RandomUniform9CNN_encoder_cat/Conv2D_2/W/Initializer/random_uniform/sub*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W
ź
5CNN_encoder_cat/Conv2D_2/W/Initializer/random_uniformAdd9CNN_encoder_cat/Conv2D_2/W/Initializer/random_uniform/mul9CNN_encoder_cat/Conv2D_2/W/Initializer/random_uniform/min*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W
„
CNN_encoder_cat/Conv2D_2/W
VariableV2*
shape:  *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0*
	container 
ß
!CNN_encoder_cat/Conv2D_2/W/AssignAssignCNN_encoder_cat/Conv2D_2/W5CNN_encoder_cat/Conv2D_2/W/Initializer/random_uniform*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
validate_shape(

CNN_encoder_cat/Conv2D_2/W/readIdentityCNN_encoder_cat/Conv2D_2/W*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W

,CNN_encoder_cat/Conv2D_2/b/Initializer/ConstConst*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
valueB *    *
dtype0

CNN_encoder_cat/Conv2D_2/b
VariableV2*
shape: *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
dtype0*
	container 
Ö
!CNN_encoder_cat/Conv2D_2/b/AssignAssignCNN_encoder_cat/Conv2D_2/b,CNN_encoder_cat/Conv2D_2/b/Initializer/Const*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
validate_shape(

CNN_encoder_cat/Conv2D_2/b/readIdentityCNN_encoder_cat/Conv2D_2/b*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b
’
CNN_encoder_cat/Conv2D_2/Conv2DConv2D#CNN_encoder_cat/MaxPool2D_1/MaxPoolCNN_encoder_cat/Conv2D_2/W/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME

 CNN_encoder_cat/Conv2D_2/BiasAddBiasAddCNN_encoder_cat/Conv2D_2/Conv2DCNN_encoder_cat/Conv2D_2/b/read*
T0*
data_formatNHWC
P
CNN_encoder_cat/Conv2D_2/TanhTanh CNN_encoder_cat/Conv2D_2/BiasAdd*
T0
Ø
#CNN_encoder_cat/MaxPool2D_2/MaxPoolMaxPoolCNN_encoder_cat/Conv2D_2/Tanh*
ksize
*
paddingSAME*
T0*
data_formatNHWC*
strides


;CNN_encoder_cat/catout/W/Initializer/truncated_normal/shapeConst*+
_class!
loc:@CNN_encoder_cat/catout/W*
valueB"     *
dtype0

:CNN_encoder_cat/catout/W/Initializer/truncated_normal/meanConst*+
_class!
loc:@CNN_encoder_cat/catout/W*
valueB
 *    *
dtype0

<CNN_encoder_cat/catout/W/Initializer/truncated_normal/stddevConst*+
_class!
loc:@CNN_encoder_cat/catout/W*
valueB
 *
×£<*
dtype0
ń
ECNN_encoder_cat/catout/W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal;CNN_encoder_cat/catout/W/Initializer/truncated_normal/shape*

seed *
T0*+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0*
seed2 
ū
9CNN_encoder_cat/catout/W/Initializer/truncated_normal/mulMulECNN_encoder_cat/catout/W/Initializer/truncated_normal/TruncatedNormal<CNN_encoder_cat/catout/W/Initializer/truncated_normal/stddev*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W
é
5CNN_encoder_cat/catout/W/Initializer/truncated_normalAdd9CNN_encoder_cat/catout/W/Initializer/truncated_normal/mul:CNN_encoder_cat/catout/W/Initializer/truncated_normal/mean*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W

CNN_encoder_cat/catout/W
VariableV2*
shape:	*
shared_name *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0*
	container 
Ł
CNN_encoder_cat/catout/W/AssignAssignCNN_encoder_cat/catout/W5CNN_encoder_cat/catout/W/Initializer/truncated_normal*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W*
validate_shape(
y
CNN_encoder_cat/catout/W/readIdentityCNN_encoder_cat/catout/W*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W

*CNN_encoder_cat/catout/b/Initializer/ConstConst*+
_class!
loc:@CNN_encoder_cat/catout/b*
valueB*    *
dtype0

CNN_encoder_cat/catout/b
VariableV2*
shape:*
shared_name *+
_class!
loc:@CNN_encoder_cat/catout/b*
dtype0*
	container 
Ī
CNN_encoder_cat/catout/b/AssignAssignCNN_encoder_cat/catout/b*CNN_encoder_cat/catout/b/Initializer/Const*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b*
validate_shape(
y
CNN_encoder_cat/catout/b/readIdentityCNN_encoder_cat/catout/b*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b
Y
$CNN_encoder_cat/catout/Reshape/shapeConst*
valueB"’’’’  *
dtype0

CNN_encoder_cat/catout/ReshapeReshape#CNN_encoder_cat/MaxPool2D_2/MaxPool$CNN_encoder_cat/catout/Reshape/shape*
T0*
Tshape0

CNN_encoder_cat/catout/MatMulMatMulCNN_encoder_cat/catout/ReshapeCNN_encoder_cat/catout/W/read*
transpose_b( *
T0*
transpose_a( 

CNN_encoder_cat/catout/BiasAddBiasAddCNN_encoder_cat/catout/MatMulCNN_encoder_cat/catout/b/read*
T0*
data_formatNHWC
R
CNN_encoder_cat/catout/SoftmaxSoftmaxCNN_encoder_cat/catout/BiasAdd*
T0

9CNN_encoder_cat/zout/W/Initializer/truncated_normal/shapeConst*)
_class
loc:@CNN_encoder_cat/zout/W*
valueB"  2   *
dtype0

8CNN_encoder_cat/zout/W/Initializer/truncated_normal/meanConst*)
_class
loc:@CNN_encoder_cat/zout/W*
valueB
 *    *
dtype0

:CNN_encoder_cat/zout/W/Initializer/truncated_normal/stddevConst*)
_class
loc:@CNN_encoder_cat/zout/W*
valueB
 *
×£<*
dtype0
ė
CCNN_encoder_cat/zout/W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal9CNN_encoder_cat/zout/W/Initializer/truncated_normal/shape*

seed *
T0*)
_class
loc:@CNN_encoder_cat/zout/W*
dtype0*
seed2 
ó
7CNN_encoder_cat/zout/W/Initializer/truncated_normal/mulMulCCNN_encoder_cat/zout/W/Initializer/truncated_normal/TruncatedNormal:CNN_encoder_cat/zout/W/Initializer/truncated_normal/stddev*
T0*)
_class
loc:@CNN_encoder_cat/zout/W
į
3CNN_encoder_cat/zout/W/Initializer/truncated_normalAdd7CNN_encoder_cat/zout/W/Initializer/truncated_normal/mul8CNN_encoder_cat/zout/W/Initializer/truncated_normal/mean*
T0*)
_class
loc:@CNN_encoder_cat/zout/W

CNN_encoder_cat/zout/W
VariableV2*
shape:	2*
shared_name *)
_class
loc:@CNN_encoder_cat/zout/W*
dtype0*
	container 
Ń
CNN_encoder_cat/zout/W/AssignAssignCNN_encoder_cat/zout/W3CNN_encoder_cat/zout/W/Initializer/truncated_normal*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/W*
validate_shape(
s
CNN_encoder_cat/zout/W/readIdentityCNN_encoder_cat/zout/W*
T0*)
_class
loc:@CNN_encoder_cat/zout/W

(CNN_encoder_cat/zout/b/Initializer/ConstConst*)
_class
loc:@CNN_encoder_cat/zout/b*
valueB2*    *
dtype0

CNN_encoder_cat/zout/b
VariableV2*
shape:2*
shared_name *)
_class
loc:@CNN_encoder_cat/zout/b*
dtype0*
	container 
Ę
CNN_encoder_cat/zout/b/AssignAssignCNN_encoder_cat/zout/b(CNN_encoder_cat/zout/b/Initializer/Const*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/b*
validate_shape(
s
CNN_encoder_cat/zout/b/readIdentityCNN_encoder_cat/zout/b*
T0*)
_class
loc:@CNN_encoder_cat/zout/b
W
"CNN_encoder_cat/zout/Reshape/shapeConst*
valueB"’’’’  *
dtype0

CNN_encoder_cat/zout/ReshapeReshape#CNN_encoder_cat/MaxPool2D_2/MaxPool"CNN_encoder_cat/zout/Reshape/shape*
T0*
Tshape0

CNN_encoder_cat/zout/MatMulMatMulCNN_encoder_cat/zout/ReshapeCNN_encoder_cat/zout/W/read*
transpose_b( *
T0*
transpose_a( 

CNN_encoder_cat/zout/BiasAddBiasAddCNN_encoder_cat/zout/MatMulCNN_encoder_cat/zout/b/read*
T0*
data_formatNHWC
5
concat/axisConst*
value	B :*
dtype0
{
concatConcatV2CNN_encoder_cat/zout/BiasAddCNN_encoder_cat/catout/Softmaxconcat/axis*

Tidx0*
T0*
N
„
?CNN_decoder/FullyConnected/W/Initializer/truncated_normal/shapeConst*/
_class%
#!loc:@CNN_decoder/FullyConnected/W*
valueB"3   Ą  *
dtype0

>CNN_decoder/FullyConnected/W/Initializer/truncated_normal/meanConst*/
_class%
#!loc:@CNN_decoder/FullyConnected/W*
valueB
 *    *
dtype0

@CNN_decoder/FullyConnected/W/Initializer/truncated_normal/stddevConst*/
_class%
#!loc:@CNN_decoder/FullyConnected/W*
valueB
 *
×£<*
dtype0
ż
ICNN_decoder/FullyConnected/W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal?CNN_decoder/FullyConnected/W/Initializer/truncated_normal/shape*

seed *
T0*/
_class%
#!loc:@CNN_decoder/FullyConnected/W*
dtype0*
seed2 

=CNN_decoder/FullyConnected/W/Initializer/truncated_normal/mulMulICNN_decoder/FullyConnected/W/Initializer/truncated_normal/TruncatedNormal@CNN_decoder/FullyConnected/W/Initializer/truncated_normal/stddev*
T0*/
_class%
#!loc:@CNN_decoder/FullyConnected/W
ł
9CNN_decoder/FullyConnected/W/Initializer/truncated_normalAdd=CNN_decoder/FullyConnected/W/Initializer/truncated_normal/mul>CNN_decoder/FullyConnected/W/Initializer/truncated_normal/mean*
T0*/
_class%
#!loc:@CNN_decoder/FullyConnected/W
¢
CNN_decoder/FullyConnected/W
VariableV2*
shape:	3Ą	*
shared_name */
_class%
#!loc:@CNN_decoder/FullyConnected/W*
dtype0*
	container 
é
#CNN_decoder/FullyConnected/W/AssignAssignCNN_decoder/FullyConnected/W9CNN_decoder/FullyConnected/W/Initializer/truncated_normal*
use_locking(*
T0*/
_class%
#!loc:@CNN_decoder/FullyConnected/W*
validate_shape(

!CNN_decoder/FullyConnected/W/readIdentityCNN_decoder/FullyConnected/W*
T0*/
_class%
#!loc:@CNN_decoder/FullyConnected/W

.CNN_decoder/FullyConnected/b/Initializer/ConstConst*/
_class%
#!loc:@CNN_decoder/FullyConnected/b*
valueBĄ	*    *
dtype0

CNN_decoder/FullyConnected/b
VariableV2*
shape:Ą	*
shared_name */
_class%
#!loc:@CNN_decoder/FullyConnected/b*
dtype0*
	container 
Ž
#CNN_decoder/FullyConnected/b/AssignAssignCNN_decoder/FullyConnected/b.CNN_decoder/FullyConnected/b/Initializer/Const*
use_locking(*
T0*/
_class%
#!loc:@CNN_decoder/FullyConnected/b*
validate_shape(

!CNN_decoder/FullyConnected/b/readIdentityCNN_decoder/FullyConnected/b*
T0*/
_class%
#!loc:@CNN_decoder/FullyConnected/b

!CNN_decoder/FullyConnected/MatMulMatMulconcat!CNN_decoder/FullyConnected/W/read*
transpose_b( *
T0*
transpose_a( 

"CNN_decoder/FullyConnected/BiasAddBiasAdd!CNN_decoder/FullyConnected/MatMul!CNN_decoder/FullyConnected/b/read*
T0*
data_formatNHWC
T
CNN_decoder/FullyConnected/TanhTanh"CNN_decoder/FullyConnected/BiasAdd*
T0
V
CNN_decoder/Reshape/shapeConst*%
valueB"’’’’   &       *
dtype0
q
CNN_decoder/ReshapeReshapeCNN_decoder/FullyConnected/TanhCNN_decoder/Reshape/shape*
T0*
Tshape0

5CNN_decoder/Conv2D/W/Initializer/random_uniform/shapeConst*'
_class
loc:@CNN_decoder/Conv2D/W*%
valueB"              *
dtype0

3CNN_decoder/Conv2D/W/Initializer/random_uniform/minConst*'
_class
loc:@CNN_decoder/Conv2D/W*
valueB
 *ģŃ½*
dtype0

3CNN_decoder/Conv2D/W/Initializer/random_uniform/maxConst*'
_class
loc:@CNN_decoder/Conv2D/W*
valueB
 *ģŃ=*
dtype0
Ż
=CNN_decoder/Conv2D/W/Initializer/random_uniform/RandomUniformRandomUniform5CNN_decoder/Conv2D/W/Initializer/random_uniform/shape*

seed *
T0*'
_class
loc:@CNN_decoder/Conv2D/W*
dtype0*
seed2 
Ö
3CNN_decoder/Conv2D/W/Initializer/random_uniform/subSub3CNN_decoder/Conv2D/W/Initializer/random_uniform/max3CNN_decoder/Conv2D/W/Initializer/random_uniform/min*
T0*'
_class
loc:@CNN_decoder/Conv2D/W
ą
3CNN_decoder/Conv2D/W/Initializer/random_uniform/mulMul=CNN_decoder/Conv2D/W/Initializer/random_uniform/RandomUniform3CNN_decoder/Conv2D/W/Initializer/random_uniform/sub*
T0*'
_class
loc:@CNN_decoder/Conv2D/W
Ņ
/CNN_decoder/Conv2D/W/Initializer/random_uniformAdd3CNN_decoder/Conv2D/W/Initializer/random_uniform/mul3CNN_decoder/Conv2D/W/Initializer/random_uniform/min*
T0*'
_class
loc:@CNN_decoder/Conv2D/W

CNN_decoder/Conv2D/W
VariableV2*
shape:  *
shared_name *'
_class
loc:@CNN_decoder/Conv2D/W*
dtype0*
	container 
Ē
CNN_decoder/Conv2D/W/AssignAssignCNN_decoder/Conv2D/W/CNN_decoder/Conv2D/W/Initializer/random_uniform*
use_locking(*
T0*'
_class
loc:@CNN_decoder/Conv2D/W*
validate_shape(
m
CNN_decoder/Conv2D/W/readIdentityCNN_decoder/Conv2D/W*
T0*'
_class
loc:@CNN_decoder/Conv2D/W

&CNN_decoder/Conv2D/b/Initializer/ConstConst*'
_class
loc:@CNN_decoder/Conv2D/b*
valueB *    *
dtype0

CNN_decoder/Conv2D/b
VariableV2*
shape: *
shared_name *'
_class
loc:@CNN_decoder/Conv2D/b*
dtype0*
	container 
¾
CNN_decoder/Conv2D/b/AssignAssignCNN_decoder/Conv2D/b&CNN_decoder/Conv2D/b/Initializer/Const*
use_locking(*
T0*'
_class
loc:@CNN_decoder/Conv2D/b*
validate_shape(
m
CNN_decoder/Conv2D/b/readIdentityCNN_decoder/Conv2D/b*
T0*'
_class
loc:@CNN_decoder/Conv2D/b
ć
CNN_decoder/Conv2D/Conv2DConv2DCNN_decoder/ReshapeCNN_decoder/Conv2D/W/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
{
CNN_decoder/Conv2D/BiasAddBiasAddCNN_decoder/Conv2D/Conv2DCNN_decoder/Conv2D/b/read*
T0*
data_formatNHWC
D
CNN_decoder/Conv2D/TanhTanhCNN_decoder/Conv2D/BiasAdd*
T0
Q
CNN_decoder/UpSample2D/ConstConst*
valueB"      *
dtype0
Q
CNN_decoder/UpSample2D/mul/xConst*
valueB"   &   *
dtype0
f
CNN_decoder/UpSample2D/mulMulCNN_decoder/UpSample2D/mul/xCNN_decoder/UpSample2D/Const*
T0
²
,CNN_decoder/UpSample2D/ResizeNearestNeighborResizeNearestNeighborCNN_decoder/Conv2D/TanhCNN_decoder/UpSample2D/mul*
align_corners( *
half_pixel_centers( *
T0

7CNN_decoder/Conv2D_1/W/Initializer/random_uniform/shapeConst*)
_class
loc:@CNN_decoder/Conv2D_1/W*%
valueB"          @   *
dtype0

5CNN_decoder/Conv2D_1/W/Initializer/random_uniform/minConst*)
_class
loc:@CNN_decoder/Conv2D_1/W*
valueB
 *ģŃ½*
dtype0

5CNN_decoder/Conv2D_1/W/Initializer/random_uniform/maxConst*)
_class
loc:@CNN_decoder/Conv2D_1/W*
valueB
 *ģŃ=*
dtype0
ć
?CNN_decoder/Conv2D_1/W/Initializer/random_uniform/RandomUniformRandomUniform7CNN_decoder/Conv2D_1/W/Initializer/random_uniform/shape*

seed *
T0*)
_class
loc:@CNN_decoder/Conv2D_1/W*
dtype0*
seed2 
Ž
5CNN_decoder/Conv2D_1/W/Initializer/random_uniform/subSub5CNN_decoder/Conv2D_1/W/Initializer/random_uniform/max5CNN_decoder/Conv2D_1/W/Initializer/random_uniform/min*
T0*)
_class
loc:@CNN_decoder/Conv2D_1/W
č
5CNN_decoder/Conv2D_1/W/Initializer/random_uniform/mulMul?CNN_decoder/Conv2D_1/W/Initializer/random_uniform/RandomUniform5CNN_decoder/Conv2D_1/W/Initializer/random_uniform/sub*
T0*)
_class
loc:@CNN_decoder/Conv2D_1/W
Ś
1CNN_decoder/Conv2D_1/W/Initializer/random_uniformAdd5CNN_decoder/Conv2D_1/W/Initializer/random_uniform/mul5CNN_decoder/Conv2D_1/W/Initializer/random_uniform/min*
T0*)
_class
loc:@CNN_decoder/Conv2D_1/W

CNN_decoder/Conv2D_1/W
VariableV2*
shape: @*
shared_name *)
_class
loc:@CNN_decoder/Conv2D_1/W*
dtype0*
	container 
Ļ
CNN_decoder/Conv2D_1/W/AssignAssignCNN_decoder/Conv2D_1/W1CNN_decoder/Conv2D_1/W/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@CNN_decoder/Conv2D_1/W*
validate_shape(
s
CNN_decoder/Conv2D_1/W/readIdentityCNN_decoder/Conv2D_1/W*
T0*)
_class
loc:@CNN_decoder/Conv2D_1/W

(CNN_decoder/Conv2D_1/b/Initializer/ConstConst*)
_class
loc:@CNN_decoder/Conv2D_1/b*
valueB@*    *
dtype0

CNN_decoder/Conv2D_1/b
VariableV2*
shape:@*
shared_name *)
_class
loc:@CNN_decoder/Conv2D_1/b*
dtype0*
	container 
Ę
CNN_decoder/Conv2D_1/b/AssignAssignCNN_decoder/Conv2D_1/b(CNN_decoder/Conv2D_1/b/Initializer/Const*
use_locking(*
T0*)
_class
loc:@CNN_decoder/Conv2D_1/b*
validate_shape(
s
CNN_decoder/Conv2D_1/b/readIdentityCNN_decoder/Conv2D_1/b*
T0*)
_class
loc:@CNN_decoder/Conv2D_1/b

CNN_decoder/Conv2D_1/Conv2DConv2D,CNN_decoder/UpSample2D/ResizeNearestNeighborCNN_decoder/Conv2D_1/W/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME

CNN_decoder/Conv2D_1/BiasAddBiasAddCNN_decoder/Conv2D_1/Conv2DCNN_decoder/Conv2D_1/b/read*
T0*
data_formatNHWC
H
CNN_decoder/Conv2D_1/TanhTanhCNN_decoder/Conv2D_1/BiasAdd*
T0
S
CNN_decoder/UpSample2D_1/ConstConst*
valueB"      *
dtype0
S
CNN_decoder/UpSample2D_1/mul/xConst*
valueB"   L   *
dtype0
l
CNN_decoder/UpSample2D_1/mulMulCNN_decoder/UpSample2D_1/mul/xCNN_decoder/UpSample2D_1/Const*
T0
ø
.CNN_decoder/UpSample2D_1/ResizeNearestNeighborResizeNearestNeighborCNN_decoder/Conv2D_1/TanhCNN_decoder/UpSample2D_1/mul*
align_corners( *
half_pixel_centers( *
T0

7CNN_decoder/Conv2D_2/W/Initializer/random_uniform/shapeConst*)
_class
loc:@CNN_decoder/Conv2D_2/W*%
valueB"      @   @   *
dtype0

5CNN_decoder/Conv2D_2/W/Initializer/random_uniform/minConst*)
_class
loc:@CNN_decoder/Conv2D_2/W*
valueB
 *:Ķ½*
dtype0

5CNN_decoder/Conv2D_2/W/Initializer/random_uniform/maxConst*)
_class
loc:@CNN_decoder/Conv2D_2/W*
valueB
 *:Ķ=*
dtype0
ć
?CNN_decoder/Conv2D_2/W/Initializer/random_uniform/RandomUniformRandomUniform7CNN_decoder/Conv2D_2/W/Initializer/random_uniform/shape*

seed *
T0*)
_class
loc:@CNN_decoder/Conv2D_2/W*
dtype0*
seed2 
Ž
5CNN_decoder/Conv2D_2/W/Initializer/random_uniform/subSub5CNN_decoder/Conv2D_2/W/Initializer/random_uniform/max5CNN_decoder/Conv2D_2/W/Initializer/random_uniform/min*
T0*)
_class
loc:@CNN_decoder/Conv2D_2/W
č
5CNN_decoder/Conv2D_2/W/Initializer/random_uniform/mulMul?CNN_decoder/Conv2D_2/W/Initializer/random_uniform/RandomUniform5CNN_decoder/Conv2D_2/W/Initializer/random_uniform/sub*
T0*)
_class
loc:@CNN_decoder/Conv2D_2/W
Ś
1CNN_decoder/Conv2D_2/W/Initializer/random_uniformAdd5CNN_decoder/Conv2D_2/W/Initializer/random_uniform/mul5CNN_decoder/Conv2D_2/W/Initializer/random_uniform/min*
T0*)
_class
loc:@CNN_decoder/Conv2D_2/W

CNN_decoder/Conv2D_2/W
VariableV2*
shape:@@*
shared_name *)
_class
loc:@CNN_decoder/Conv2D_2/W*
dtype0*
	container 
Ļ
CNN_decoder/Conv2D_2/W/AssignAssignCNN_decoder/Conv2D_2/W1CNN_decoder/Conv2D_2/W/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@CNN_decoder/Conv2D_2/W*
validate_shape(
s
CNN_decoder/Conv2D_2/W/readIdentityCNN_decoder/Conv2D_2/W*
T0*)
_class
loc:@CNN_decoder/Conv2D_2/W

(CNN_decoder/Conv2D_2/b/Initializer/ConstConst*)
_class
loc:@CNN_decoder/Conv2D_2/b*
valueB@*    *
dtype0

CNN_decoder/Conv2D_2/b
VariableV2*
shape:@*
shared_name *)
_class
loc:@CNN_decoder/Conv2D_2/b*
dtype0*
	container 
Ę
CNN_decoder/Conv2D_2/b/AssignAssignCNN_decoder/Conv2D_2/b(CNN_decoder/Conv2D_2/b/Initializer/Const*
use_locking(*
T0*)
_class
loc:@CNN_decoder/Conv2D_2/b*
validate_shape(
s
CNN_decoder/Conv2D_2/b/readIdentityCNN_decoder/Conv2D_2/b*
T0*)
_class
loc:@CNN_decoder/Conv2D_2/b

CNN_decoder/Conv2D_2/Conv2DConv2D.CNN_decoder/UpSample2D_1/ResizeNearestNeighborCNN_decoder/Conv2D_2/W/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME

CNN_decoder/Conv2D_2/BiasAddBiasAddCNN_decoder/Conv2D_2/Conv2DCNN_decoder/Conv2D_2/b/read*
T0*
data_formatNHWC
H
CNN_decoder/Conv2D_2/TanhTanhCNN_decoder/Conv2D_2/BiasAdd*
T0
S
CNN_decoder/UpSample2D_2/ConstConst*
valueB"      *
dtype0
S
CNN_decoder/UpSample2D_2/mul/xConst*
valueB"      *
dtype0
l
CNN_decoder/UpSample2D_2/mulMulCNN_decoder/UpSample2D_2/mul/xCNN_decoder/UpSample2D_2/Const*
T0
ø
.CNN_decoder/UpSample2D_2/ResizeNearestNeighborResizeNearestNeighborCNN_decoder/Conv2D_2/TanhCNN_decoder/UpSample2D_2/mul*
align_corners( *
half_pixel_centers( *
T0

5CNN_decoder/sigout/W/Initializer/random_uniform/shapeConst*'
_class
loc:@CNN_decoder/sigout/W*%
valueB"      @      *
dtype0

3CNN_decoder/sigout/W/Initializer/random_uniform/minConst*'
_class
loc:@CNN_decoder/sigout/W*
valueB
 *:Ķ½*
dtype0

3CNN_decoder/sigout/W/Initializer/random_uniform/maxConst*'
_class
loc:@CNN_decoder/sigout/W*
valueB
 *:Ķ=*
dtype0
Ż
=CNN_decoder/sigout/W/Initializer/random_uniform/RandomUniformRandomUniform5CNN_decoder/sigout/W/Initializer/random_uniform/shape*

seed *
T0*'
_class
loc:@CNN_decoder/sigout/W*
dtype0*
seed2 
Ö
3CNN_decoder/sigout/W/Initializer/random_uniform/subSub3CNN_decoder/sigout/W/Initializer/random_uniform/max3CNN_decoder/sigout/W/Initializer/random_uniform/min*
T0*'
_class
loc:@CNN_decoder/sigout/W
ą
3CNN_decoder/sigout/W/Initializer/random_uniform/mulMul=CNN_decoder/sigout/W/Initializer/random_uniform/RandomUniform3CNN_decoder/sigout/W/Initializer/random_uniform/sub*
T0*'
_class
loc:@CNN_decoder/sigout/W
Ņ
/CNN_decoder/sigout/W/Initializer/random_uniformAdd3CNN_decoder/sigout/W/Initializer/random_uniform/mul3CNN_decoder/sigout/W/Initializer/random_uniform/min*
T0*'
_class
loc:@CNN_decoder/sigout/W

CNN_decoder/sigout/W
VariableV2*
shape:@*
shared_name *'
_class
loc:@CNN_decoder/sigout/W*
dtype0*
	container 
Ē
CNN_decoder/sigout/W/AssignAssignCNN_decoder/sigout/W/CNN_decoder/sigout/W/Initializer/random_uniform*
use_locking(*
T0*'
_class
loc:@CNN_decoder/sigout/W*
validate_shape(
m
CNN_decoder/sigout/W/readIdentityCNN_decoder/sigout/W*
T0*'
_class
loc:@CNN_decoder/sigout/W

&CNN_decoder/sigout/b/Initializer/ConstConst*'
_class
loc:@CNN_decoder/sigout/b*
valueB*    *
dtype0

CNN_decoder/sigout/b
VariableV2*
shape:*
shared_name *'
_class
loc:@CNN_decoder/sigout/b*
dtype0*
	container 
¾
CNN_decoder/sigout/b/AssignAssignCNN_decoder/sigout/b&CNN_decoder/sigout/b/Initializer/Const*
use_locking(*
T0*'
_class
loc:@CNN_decoder/sigout/b*
validate_shape(
m
CNN_decoder/sigout/b/readIdentityCNN_decoder/sigout/b*
T0*'
_class
loc:@CNN_decoder/sigout/b
ž
CNN_decoder/sigout/Conv2DConv2D.CNN_decoder/UpSample2D_2/ResizeNearestNeighborCNN_decoder/sigout/W/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
{
CNN_decoder/sigout/BiasAddBiasAddCNN_decoder/sigout/Conv2DCNN_decoder/sigout/b/read*
T0*
data_formatNHWC
D
CNN_decoder/sigout/TanhTanhCNN_decoder/sigout/BiasAdd*
T0

9CNN_decoder/sigout_1/W/Initializer/truncated_normal/shapeConst*)
_class
loc:@CNN_decoder/sigout_1/W*
valueB"      *
dtype0

8CNN_decoder/sigout_1/W/Initializer/truncated_normal/meanConst*)
_class
loc:@CNN_decoder/sigout_1/W*
valueB
 *    *
dtype0

:CNN_decoder/sigout_1/W/Initializer/truncated_normal/stddevConst*)
_class
loc:@CNN_decoder/sigout_1/W*
valueB
 *
×£<*
dtype0
ė
CCNN_decoder/sigout_1/W/Initializer/truncated_normal/TruncatedNormalTruncatedNormal9CNN_decoder/sigout_1/W/Initializer/truncated_normal/shape*

seed *
T0*)
_class
loc:@CNN_decoder/sigout_1/W*
dtype0*
seed2 
ó
7CNN_decoder/sigout_1/W/Initializer/truncated_normal/mulMulCCNN_decoder/sigout_1/W/Initializer/truncated_normal/TruncatedNormal:CNN_decoder/sigout_1/W/Initializer/truncated_normal/stddev*
T0*)
_class
loc:@CNN_decoder/sigout_1/W
į
3CNN_decoder/sigout_1/W/Initializer/truncated_normalAdd7CNN_decoder/sigout_1/W/Initializer/truncated_normal/mul8CNN_decoder/sigout_1/W/Initializer/truncated_normal/mean*
T0*)
_class
loc:@CNN_decoder/sigout_1/W

CNN_decoder/sigout_1/W
VariableV2*
shape:
  *
shared_name *)
_class
loc:@CNN_decoder/sigout_1/W*
dtype0*
	container 
Ń
CNN_decoder/sigout_1/W/AssignAssignCNN_decoder/sigout_1/W3CNN_decoder/sigout_1/W/Initializer/truncated_normal*
use_locking(*
T0*)
_class
loc:@CNN_decoder/sigout_1/W*
validate_shape(
s
CNN_decoder/sigout_1/W/readIdentityCNN_decoder/sigout_1/W*
T0*)
_class
loc:@CNN_decoder/sigout_1/W

(CNN_decoder/sigout_1/b/Initializer/ConstConst*)
_class
loc:@CNN_decoder/sigout_1/b*
valueB *    *
dtype0

CNN_decoder/sigout_1/b
VariableV2*
shape: *
shared_name *)
_class
loc:@CNN_decoder/sigout_1/b*
dtype0*
	container 
Ę
CNN_decoder/sigout_1/b/AssignAssignCNN_decoder/sigout_1/b(CNN_decoder/sigout_1/b/Initializer/Const*
use_locking(*
T0*)
_class
loc:@CNN_decoder/sigout_1/b*
validate_shape(
s
CNN_decoder/sigout_1/b/readIdentityCNN_decoder/sigout_1/b*
T0*)
_class
loc:@CNN_decoder/sigout_1/b
W
"CNN_decoder/sigout_1/Reshape/shapeConst*
valueB"’’’’   *
dtype0
{
CNN_decoder/sigout_1/ReshapeReshapeCNN_decoder/sigout/Tanh"CNN_decoder/sigout_1/Reshape/shape*
T0*
Tshape0

CNN_decoder/sigout_1/MatMulMatMulCNN_decoder/sigout_1/ReshapeCNN_decoder/sigout_1/W/read*
transpose_b( *
T0*
transpose_a( 

CNN_decoder/sigout_1/BiasAddBiasAddCNN_decoder/sigout_1/MatMulCNN_decoder/sigout_1/b/read*
T0*
data_formatNHWC
[
"CNN_decoder/reshaped/Reshape/shapeConst*!
valueB"’’’’
     *
dtype0

CNN_decoder/reshaped/ReshapeReshapeCNN_decoder/sigout_1/BiasAdd"CNN_decoder/reshaped/Reshape/shape*
T0*
Tshape0
Y
SquaredDifferenceSquaredDifference
target_imgCNN_decoder/reshaped/Reshape*
T0
J
Sum/reduction_indicesConst*
valueB"      *
dtype0
Z
SumSumSquaredDifferenceSum/reduction_indices*

Tidx0*
	keep_dims( *
T0
3
ConstConst*
valueB: *
dtype0
>
MeanMeanSumConst*

Tidx0*
	keep_dims( *
T0

1discriminator/w0/Initializer/random_uniform/shapeConst*#
_class
loc:@discriminator/w0*
valueB"2      *
dtype0

/discriminator/w0/Initializer/random_uniform/minConst*#
_class
loc:@discriminator/w0*
valueB
 *c¾*
dtype0

/discriminator/w0/Initializer/random_uniform/maxConst*#
_class
loc:@discriminator/w0*
valueB
 *c>*
dtype0
Ń
9discriminator/w0/Initializer/random_uniform/RandomUniformRandomUniform1discriminator/w0/Initializer/random_uniform/shape*

seed *
T0*#
_class
loc:@discriminator/w0*
dtype0*
seed2 
Ę
/discriminator/w0/Initializer/random_uniform/subSub/discriminator/w0/Initializer/random_uniform/max/discriminator/w0/Initializer/random_uniform/min*
T0*#
_class
loc:@discriminator/w0
Š
/discriminator/w0/Initializer/random_uniform/mulMul9discriminator/w0/Initializer/random_uniform/RandomUniform/discriminator/w0/Initializer/random_uniform/sub*
T0*#
_class
loc:@discriminator/w0
Ā
+discriminator/w0/Initializer/random_uniformAdd/discriminator/w0/Initializer/random_uniform/mul/discriminator/w0/Initializer/random_uniform/min*
T0*#
_class
loc:@discriminator/w0

discriminator/w0
VariableV2*
shape:	2*
shared_name *#
_class
loc:@discriminator/w0*
dtype0*
	container 
·
discriminator/w0/AssignAssigndiscriminator/w0+discriminator/w0/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@discriminator/w0*
validate_shape(
a
discriminator/w0/readIdentitydiscriminator/w0*
T0*#
_class
loc:@discriminator/w0
y
"discriminator/b0/Initializer/ConstConst*#
_class
loc:@discriminator/b0*
valueB*    *
dtype0

discriminator/b0
VariableV2*
shape:*
shared_name *#
_class
loc:@discriminator/b0*
dtype0*
	container 
®
discriminator/b0/AssignAssigndiscriminator/b0"discriminator/b0/Initializer/Const*
use_locking(*
T0*#
_class
loc:@discriminator/b0*
validate_shape(
a
discriminator/b0/readIdentitydiscriminator/b0*
T0*#
_class
loc:@discriminator/b0
r
discriminator/MatMulMatMulprior_samplediscriminator/w0/read*
transpose_b( *
T0*
transpose_a( 
P
discriminator/addAddV2discriminator/MatMuldiscriminator/b0/read*
T0
6
discriminator/ReluReludiscriminator/add*
T0
@
discriminator/sub/xConst*
valueB
 *  ?*
dtype0
A
discriminator/subSubdiscriminator/sub/x	keep_prob*
T0
Q
discriminator/dropout/ShapeShapediscriminator/Relu*
T0*
out_type0
U
(discriminator/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
U
(discriminator/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0

2discriminator/dropout/random_uniform/RandomUniformRandomUniformdiscriminator/dropout/Shape*

seed *
T0*
dtype0*
seed2 

(discriminator/dropout/random_uniform/subSub(discriminator/dropout/random_uniform/max(discriminator/dropout/random_uniform/min*
T0

(discriminator/dropout/random_uniform/mulMul2discriminator/dropout/random_uniform/RandomUniform(discriminator/dropout/random_uniform/sub*
T0

$discriminator/dropout/random_uniformAdd(discriminator/dropout/random_uniform/mul(discriminator/dropout/random_uniform/min*
T0
H
discriminator/dropout/sub/xConst*
valueB
 *  ?*
dtype0
Y
discriminator/dropout/subSubdiscriminator/dropout/sub/xdiscriminator/sub*
T0
L
discriminator/dropout/truediv/xConst*
valueB
 *  ?*
dtype0
m
discriminator/dropout/truedivRealDivdiscriminator/dropout/truediv/xdiscriminator/dropout/sub*
T0
t
"discriminator/dropout/GreaterEqualGreaterEqual$discriminator/dropout/random_uniformdiscriminator/sub*
T0
\
discriminator/dropout/mulMuldiscriminator/Reludiscriminator/dropout/truediv*
T0
n
discriminator/dropout/CastCast"discriminator/dropout/GreaterEqual*

SrcT0
*
Truncate( *

DstT0
b
discriminator/dropout/mul_1Muldiscriminator/dropout/muldiscriminator/dropout/Cast*
T0

1discriminator/w1/Initializer/random_uniform/shapeConst*#
_class
loc:@discriminator/w1*
valueB"      *
dtype0

/discriminator/w1/Initializer/random_uniform/minConst*#
_class
loc:@discriminator/w1*
valueB
 *×³Ż½*
dtype0

/discriminator/w1/Initializer/random_uniform/maxConst*#
_class
loc:@discriminator/w1*
valueB
 *×³Ż=*
dtype0
Ń
9discriminator/w1/Initializer/random_uniform/RandomUniformRandomUniform1discriminator/w1/Initializer/random_uniform/shape*

seed *
T0*#
_class
loc:@discriminator/w1*
dtype0*
seed2 
Ę
/discriminator/w1/Initializer/random_uniform/subSub/discriminator/w1/Initializer/random_uniform/max/discriminator/w1/Initializer/random_uniform/min*
T0*#
_class
loc:@discriminator/w1
Š
/discriminator/w1/Initializer/random_uniform/mulMul9discriminator/w1/Initializer/random_uniform/RandomUniform/discriminator/w1/Initializer/random_uniform/sub*
T0*#
_class
loc:@discriminator/w1
Ā
+discriminator/w1/Initializer/random_uniformAdd/discriminator/w1/Initializer/random_uniform/mul/discriminator/w1/Initializer/random_uniform/min*
T0*#
_class
loc:@discriminator/w1

discriminator/w1
VariableV2*
shape:
*
shared_name *#
_class
loc:@discriminator/w1*
dtype0*
	container 
·
discriminator/w1/AssignAssigndiscriminator/w1+discriminator/w1/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@discriminator/w1*
validate_shape(
a
discriminator/w1/readIdentitydiscriminator/w1*
T0*#
_class
loc:@discriminator/w1
y
"discriminator/b1/Initializer/ConstConst*#
_class
loc:@discriminator/b1*
valueB*    *
dtype0

discriminator/b1
VariableV2*
shape:*
shared_name *#
_class
loc:@discriminator/b1*
dtype0*
	container 
®
discriminator/b1/AssignAssigndiscriminator/b1"discriminator/b1/Initializer/Const*
use_locking(*
T0*#
_class
loc:@discriminator/b1*
validate_shape(
a
discriminator/b1/readIdentitydiscriminator/b1*
T0*#
_class
loc:@discriminator/b1

discriminator/MatMul_1MatMuldiscriminator/dropout/mul_1discriminator/w1/read*
transpose_b( *
T0*
transpose_a( 
T
discriminator/add_1AddV2discriminator/MatMul_1discriminator/b1/read*
T0
:
discriminator/Relu_1Reludiscriminator/add_1*
T0
B
discriminator/sub_1/xConst*
valueB
 *  ?*
dtype0
E
discriminator/sub_1Subdiscriminator/sub_1/x	keep_prob*
T0
U
discriminator/dropout_1/ShapeShapediscriminator/Relu_1*
T0*
out_type0
W
*discriminator/dropout_1/random_uniform/minConst*
valueB
 *    *
dtype0
W
*discriminator/dropout_1/random_uniform/maxConst*
valueB
 *  ?*
dtype0

4discriminator/dropout_1/random_uniform/RandomUniformRandomUniformdiscriminator/dropout_1/Shape*

seed *
T0*
dtype0*
seed2 

*discriminator/dropout_1/random_uniform/subSub*discriminator/dropout_1/random_uniform/max*discriminator/dropout_1/random_uniform/min*
T0

*discriminator/dropout_1/random_uniform/mulMul4discriminator/dropout_1/random_uniform/RandomUniform*discriminator/dropout_1/random_uniform/sub*
T0

&discriminator/dropout_1/random_uniformAdd*discriminator/dropout_1/random_uniform/mul*discriminator/dropout_1/random_uniform/min*
T0
J
discriminator/dropout_1/sub/xConst*
valueB
 *  ?*
dtype0
_
discriminator/dropout_1/subSubdiscriminator/dropout_1/sub/xdiscriminator/sub_1*
T0
N
!discriminator/dropout_1/truediv/xConst*
valueB
 *  ?*
dtype0
s
discriminator/dropout_1/truedivRealDiv!discriminator/dropout_1/truediv/xdiscriminator/dropout_1/sub*
T0
z
$discriminator/dropout_1/GreaterEqualGreaterEqual&discriminator/dropout_1/random_uniformdiscriminator/sub_1*
T0
b
discriminator/dropout_1/mulMuldiscriminator/Relu_1discriminator/dropout_1/truediv*
T0
r
discriminator/dropout_1/CastCast$discriminator/dropout_1/GreaterEqual*

SrcT0
*
Truncate( *

DstT0
h
discriminator/dropout_1/mul_1Muldiscriminator/dropout_1/muldiscriminator/dropout_1/Cast*
T0

1discriminator/wo/Initializer/random_uniform/shapeConst*#
_class
loc:@discriminator/wo*
valueB"      *
dtype0

/discriminator/wo/Initializer/random_uniform/minConst*#
_class
loc:@discriminator/wo*
valueB
 *Iv¾*
dtype0

/discriminator/wo/Initializer/random_uniform/maxConst*#
_class
loc:@discriminator/wo*
valueB
 *Iv>*
dtype0
Ń
9discriminator/wo/Initializer/random_uniform/RandomUniformRandomUniform1discriminator/wo/Initializer/random_uniform/shape*

seed *
T0*#
_class
loc:@discriminator/wo*
dtype0*
seed2 
Ę
/discriminator/wo/Initializer/random_uniform/subSub/discriminator/wo/Initializer/random_uniform/max/discriminator/wo/Initializer/random_uniform/min*
T0*#
_class
loc:@discriminator/wo
Š
/discriminator/wo/Initializer/random_uniform/mulMul9discriminator/wo/Initializer/random_uniform/RandomUniform/discriminator/wo/Initializer/random_uniform/sub*
T0*#
_class
loc:@discriminator/wo
Ā
+discriminator/wo/Initializer/random_uniformAdd/discriminator/wo/Initializer/random_uniform/mul/discriminator/wo/Initializer/random_uniform/min*
T0*#
_class
loc:@discriminator/wo

discriminator/wo
VariableV2*
shape:	*
shared_name *#
_class
loc:@discriminator/wo*
dtype0*
	container 
·
discriminator/wo/AssignAssigndiscriminator/wo+discriminator/wo/Initializer/random_uniform*
use_locking(*
T0*#
_class
loc:@discriminator/wo*
validate_shape(
a
discriminator/wo/readIdentitydiscriminator/wo*
T0*#
_class
loc:@discriminator/wo
x
"discriminator/bo/Initializer/ConstConst*#
_class
loc:@discriminator/bo*
valueB*    *
dtype0

discriminator/bo
VariableV2*
shape:*
shared_name *#
_class
loc:@discriminator/bo*
dtype0*
	container 
®
discriminator/bo/AssignAssigndiscriminator/bo"discriminator/bo/Initializer/Const*
use_locking(*
T0*#
_class
loc:@discriminator/bo*
validate_shape(
a
discriminator/bo/readIdentitydiscriminator/bo*
T0*#
_class
loc:@discriminator/bo

discriminator/MatMul_2MatMuldiscriminator/dropout_1/mul_1discriminator/wo/read*
transpose_b( *
T0*
transpose_a( 
T
discriminator/add_2AddV2discriminator/MatMul_2discriminator/bo/read*
T0
0
SigmoidSigmoiddiscriminator/add_2*
T0

discriminator_1/MatMulMatMulCNN_encoder_cat/zout/BiasAdddiscriminator/w0/read*
transpose_b( *
T0*
transpose_a( 
T
discriminator_1/addAddV2discriminator_1/MatMuldiscriminator/b0/read*
T0
:
discriminator_1/ReluReludiscriminator_1/add*
T0
B
discriminator_1/sub/xConst*
valueB
 *  ?*
dtype0
E
discriminator_1/subSubdiscriminator_1/sub/x	keep_prob*
T0
U
discriminator_1/dropout/ShapeShapediscriminator_1/Relu*
T0*
out_type0
W
*discriminator_1/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
W
*discriminator_1/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0

4discriminator_1/dropout/random_uniform/RandomUniformRandomUniformdiscriminator_1/dropout/Shape*

seed *
T0*
dtype0*
seed2 

*discriminator_1/dropout/random_uniform/subSub*discriminator_1/dropout/random_uniform/max*discriminator_1/dropout/random_uniform/min*
T0

*discriminator_1/dropout/random_uniform/mulMul4discriminator_1/dropout/random_uniform/RandomUniform*discriminator_1/dropout/random_uniform/sub*
T0

&discriminator_1/dropout/random_uniformAdd*discriminator_1/dropout/random_uniform/mul*discriminator_1/dropout/random_uniform/min*
T0
J
discriminator_1/dropout/sub/xConst*
valueB
 *  ?*
dtype0
_
discriminator_1/dropout/subSubdiscriminator_1/dropout/sub/xdiscriminator_1/sub*
T0
N
!discriminator_1/dropout/truediv/xConst*
valueB
 *  ?*
dtype0
s
discriminator_1/dropout/truedivRealDiv!discriminator_1/dropout/truediv/xdiscriminator_1/dropout/sub*
T0
z
$discriminator_1/dropout/GreaterEqualGreaterEqual&discriminator_1/dropout/random_uniformdiscriminator_1/sub*
T0
b
discriminator_1/dropout/mulMuldiscriminator_1/Reludiscriminator_1/dropout/truediv*
T0
r
discriminator_1/dropout/CastCast$discriminator_1/dropout/GreaterEqual*

SrcT0
*
Truncate( *

DstT0
h
discriminator_1/dropout/mul_1Muldiscriminator_1/dropout/muldiscriminator_1/dropout/Cast*
T0

discriminator_1/MatMul_1MatMuldiscriminator_1/dropout/mul_1discriminator/w1/read*
transpose_b( *
T0*
transpose_a( 
X
discriminator_1/add_1AddV2discriminator_1/MatMul_1discriminator/b1/read*
T0
>
discriminator_1/Relu_1Reludiscriminator_1/add_1*
T0
D
discriminator_1/sub_1/xConst*
valueB
 *  ?*
dtype0
I
discriminator_1/sub_1Subdiscriminator_1/sub_1/x	keep_prob*
T0
Y
discriminator_1/dropout_1/ShapeShapediscriminator_1/Relu_1*
T0*
out_type0
Y
,discriminator_1/dropout_1/random_uniform/minConst*
valueB
 *    *
dtype0
Y
,discriminator_1/dropout_1/random_uniform/maxConst*
valueB
 *  ?*
dtype0

6discriminator_1/dropout_1/random_uniform/RandomUniformRandomUniformdiscriminator_1/dropout_1/Shape*

seed *
T0*
dtype0*
seed2 

,discriminator_1/dropout_1/random_uniform/subSub,discriminator_1/dropout_1/random_uniform/max,discriminator_1/dropout_1/random_uniform/min*
T0
¢
,discriminator_1/dropout_1/random_uniform/mulMul6discriminator_1/dropout_1/random_uniform/RandomUniform,discriminator_1/dropout_1/random_uniform/sub*
T0

(discriminator_1/dropout_1/random_uniformAdd,discriminator_1/dropout_1/random_uniform/mul,discriminator_1/dropout_1/random_uniform/min*
T0
L
discriminator_1/dropout_1/sub/xConst*
valueB
 *  ?*
dtype0
e
discriminator_1/dropout_1/subSubdiscriminator_1/dropout_1/sub/xdiscriminator_1/sub_1*
T0
P
#discriminator_1/dropout_1/truediv/xConst*
valueB
 *  ?*
dtype0
y
!discriminator_1/dropout_1/truedivRealDiv#discriminator_1/dropout_1/truediv/xdiscriminator_1/dropout_1/sub*
T0

&discriminator_1/dropout_1/GreaterEqualGreaterEqual(discriminator_1/dropout_1/random_uniformdiscriminator_1/sub_1*
T0
h
discriminator_1/dropout_1/mulMuldiscriminator_1/Relu_1!discriminator_1/dropout_1/truediv*
T0
v
discriminator_1/dropout_1/CastCast&discriminator_1/dropout_1/GreaterEqual*

SrcT0
*
Truncate( *

DstT0
n
discriminator_1/dropout_1/mul_1Muldiscriminator_1/dropout_1/muldiscriminator_1/dropout_1/Cast*
T0

discriminator_1/MatMul_2MatMuldiscriminator_1/dropout_1/mul_1discriminator/wo/read*
transpose_b( *
T0*
transpose_a( 
X
discriminator_1/add_2AddV2discriminator_1/MatMul_2discriminator/bo/read*
T0
4
	Sigmoid_1Sigmoiddiscriminator_1/add_2*
T0
F
ones_like/ShapeShapediscriminator/add_2*
T0*
out_type0
<
ones_like/ConstConst*
valueB
 *  ?*
dtype0
N
	ones_likeFillones_like/Shapeones_like/Const*
T0*

index_type0
C
logistic_loss/zeros_like	ZerosLikediscriminator/add_2*
T0
b
logistic_loss/GreaterEqualGreaterEqualdiscriminator/add_2logistic_loss/zeros_like*
T0
r
logistic_loss/SelectSelectlogistic_loss/GreaterEqualdiscriminator/add_2logistic_loss/zeros_like*
T0
6
logistic_loss/NegNegdiscriminator/add_2*
T0
m
logistic_loss/Select_1Selectlogistic_loss/GreaterEquallogistic_loss/Negdiscriminator/add_2*
T0
A
logistic_loss/mulMuldiscriminator/add_2	ones_like*
T0
J
logistic_loss/subSublogistic_loss/Selectlogistic_loss/mul*
T0
9
logistic_loss/ExpExplogistic_loss/Select_1*
T0
8
logistic_loss/Log1pLog1plogistic_loss/Exp*
T0
E
logistic_lossAddlogistic_loss/sublogistic_loss/Log1p*
T0
<
Const_1Const*
valueB"       *
dtype0
L
Mean_1Meanlogistic_lossConst_1*

Tidx0*
	keep_dims( *
T0
7

zeros_like	ZerosLikediscriminator_1/add_2*
T0
G
logistic_loss_1/zeros_like	ZerosLikediscriminator_1/add_2*
T0
h
logistic_loss_1/GreaterEqualGreaterEqualdiscriminator_1/add_2logistic_loss_1/zeros_like*
T0
z
logistic_loss_1/SelectSelectlogistic_loss_1/GreaterEqualdiscriminator_1/add_2logistic_loss_1/zeros_like*
T0
:
logistic_loss_1/NegNegdiscriminator_1/add_2*
T0
u
logistic_loss_1/Select_1Selectlogistic_loss_1/GreaterEquallogistic_loss_1/Negdiscriminator_1/add_2*
T0
F
logistic_loss_1/mulMuldiscriminator_1/add_2
zeros_like*
T0
P
logistic_loss_1/subSublogistic_loss_1/Selectlogistic_loss_1/mul*
T0
=
logistic_loss_1/ExpExplogistic_loss_1/Select_1*
T0
<
logistic_loss_1/Log1pLog1plogistic_loss_1/Exp*
T0
K
logistic_loss_1Addlogistic_loss_1/sublogistic_loss_1/Log1p*
T0
<
Const_2Const*
valueB"       *
dtype0
N
Mean_2Meanlogistic_loss_1Const_2*

Tidx0*
	keep_dims( *
T0
%
addAddV2Mean_1Mean_2*
T0
J
ones_like_1/ShapeShapediscriminator_1/add_2*
T0*
out_type0
>
ones_like_1/ConstConst*
valueB
 *  ?*
dtype0
T
ones_like_1Fillones_like_1/Shapeones_like_1/Const*
T0*

index_type0
G
logistic_loss_2/zeros_like	ZerosLikediscriminator_1/add_2*
T0
h
logistic_loss_2/GreaterEqualGreaterEqualdiscriminator_1/add_2logistic_loss_2/zeros_like*
T0
z
logistic_loss_2/SelectSelectlogistic_loss_2/GreaterEqualdiscriminator_1/add_2logistic_loss_2/zeros_like*
T0
:
logistic_loss_2/NegNegdiscriminator_1/add_2*
T0
u
logistic_loss_2/Select_1Selectlogistic_loss_2/GreaterEquallogistic_loss_2/Negdiscriminator_1/add_2*
T0
G
logistic_loss_2/mulMuldiscriminator_1/add_2ones_like_1*
T0
P
logistic_loss_2/subSublogistic_loss_2/Selectlogistic_loss_2/mul*
T0
=
logistic_loss_2/ExpExplogistic_loss_2/Select_1*
T0
<
logistic_loss_2/Log1pLog1plogistic_loss_2/Exp*
T0
K
logistic_loss_2Addlogistic_loss_2/sublogistic_loss_2/Log1p*
T0
<
Const_3Const*
valueB"       *
dtype0
N
Mean_3Meanlogistic_loss_2Const_3*

Tidx0*
	keep_dims( *
T0

5discriminator_cat/w0/Initializer/random_uniform/shapeConst*'
_class
loc:@discriminator_cat/w0*
valueB"      *
dtype0

3discriminator_cat/w0/Initializer/random_uniform/minConst*'
_class
loc:@discriminator_cat/w0*
valueB
 *Iv¾*
dtype0

3discriminator_cat/w0/Initializer/random_uniform/maxConst*'
_class
loc:@discriminator_cat/w0*
valueB
 *Iv>*
dtype0
Ż
=discriminator_cat/w0/Initializer/random_uniform/RandomUniformRandomUniform5discriminator_cat/w0/Initializer/random_uniform/shape*

seed *
T0*'
_class
loc:@discriminator_cat/w0*
dtype0*
seed2 
Ö
3discriminator_cat/w0/Initializer/random_uniform/subSub3discriminator_cat/w0/Initializer/random_uniform/max3discriminator_cat/w0/Initializer/random_uniform/min*
T0*'
_class
loc:@discriminator_cat/w0
ą
3discriminator_cat/w0/Initializer/random_uniform/mulMul=discriminator_cat/w0/Initializer/random_uniform/RandomUniform3discriminator_cat/w0/Initializer/random_uniform/sub*
T0*'
_class
loc:@discriminator_cat/w0
Ņ
/discriminator_cat/w0/Initializer/random_uniformAdd3discriminator_cat/w0/Initializer/random_uniform/mul3discriminator_cat/w0/Initializer/random_uniform/min*
T0*'
_class
loc:@discriminator_cat/w0

discriminator_cat/w0
VariableV2*
shape:	*
shared_name *'
_class
loc:@discriminator_cat/w0*
dtype0*
	container 
Ē
discriminator_cat/w0/AssignAssigndiscriminator_cat/w0/discriminator_cat/w0/Initializer/random_uniform*
use_locking(*
T0*'
_class
loc:@discriminator_cat/w0*
validate_shape(
m
discriminator_cat/w0/readIdentitydiscriminator_cat/w0*
T0*'
_class
loc:@discriminator_cat/w0

&discriminator_cat/b0/Initializer/ConstConst*'
_class
loc:@discriminator_cat/b0*
valueB*    *
dtype0

discriminator_cat/b0
VariableV2*
shape:*
shared_name *'
_class
loc:@discriminator_cat/b0*
dtype0*
	container 
¾
discriminator_cat/b0/AssignAssigndiscriminator_cat/b0&discriminator_cat/b0/Initializer/Const*
use_locking(*
T0*'
_class
loc:@discriminator_cat/b0*
validate_shape(
m
discriminator_cat/b0/readIdentitydiscriminator_cat/b0*
T0*'
_class
loc:@discriminator_cat/b0

discriminator_cat/MatMulMatMulprior_sample_labeldiscriminator_cat/w0/read*
transpose_b( *
T0*
transpose_a( 
\
discriminator_cat/addAddV2discriminator_cat/MatMuldiscriminator_cat/b0/read*
T0
>
discriminator_cat/ReluReludiscriminator_cat/add*
T0
D
discriminator_cat/sub/xConst*
valueB
 *  ?*
dtype0
I
discriminator_cat/subSubdiscriminator_cat/sub/x	keep_prob*
T0
Y
discriminator_cat/dropout/ShapeShapediscriminator_cat/Relu*
T0*
out_type0
Y
,discriminator_cat/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
Y
,discriminator_cat/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0

6discriminator_cat/dropout/random_uniform/RandomUniformRandomUniformdiscriminator_cat/dropout/Shape*

seed *
T0*
dtype0*
seed2 

,discriminator_cat/dropout/random_uniform/subSub,discriminator_cat/dropout/random_uniform/max,discriminator_cat/dropout/random_uniform/min*
T0
¢
,discriminator_cat/dropout/random_uniform/mulMul6discriminator_cat/dropout/random_uniform/RandomUniform,discriminator_cat/dropout/random_uniform/sub*
T0

(discriminator_cat/dropout/random_uniformAdd,discriminator_cat/dropout/random_uniform/mul,discriminator_cat/dropout/random_uniform/min*
T0
L
discriminator_cat/dropout/sub/xConst*
valueB
 *  ?*
dtype0
e
discriminator_cat/dropout/subSubdiscriminator_cat/dropout/sub/xdiscriminator_cat/sub*
T0
P
#discriminator_cat/dropout/truediv/xConst*
valueB
 *  ?*
dtype0
y
!discriminator_cat/dropout/truedivRealDiv#discriminator_cat/dropout/truediv/xdiscriminator_cat/dropout/sub*
T0

&discriminator_cat/dropout/GreaterEqualGreaterEqual(discriminator_cat/dropout/random_uniformdiscriminator_cat/sub*
T0
h
discriminator_cat/dropout/mulMuldiscriminator_cat/Relu!discriminator_cat/dropout/truediv*
T0
v
discriminator_cat/dropout/CastCast&discriminator_cat/dropout/GreaterEqual*

SrcT0
*
Truncate( *

DstT0
n
discriminator_cat/dropout/mul_1Muldiscriminator_cat/dropout/muldiscriminator_cat/dropout/Cast*
T0

5discriminator_cat/w1/Initializer/random_uniform/shapeConst*'
_class
loc:@discriminator_cat/w1*
valueB"      *
dtype0

3discriminator_cat/w1/Initializer/random_uniform/minConst*'
_class
loc:@discriminator_cat/w1*
valueB
 *×³Ż½*
dtype0

3discriminator_cat/w1/Initializer/random_uniform/maxConst*'
_class
loc:@discriminator_cat/w1*
valueB
 *×³Ż=*
dtype0
Ż
=discriminator_cat/w1/Initializer/random_uniform/RandomUniformRandomUniform5discriminator_cat/w1/Initializer/random_uniform/shape*

seed *
T0*'
_class
loc:@discriminator_cat/w1*
dtype0*
seed2 
Ö
3discriminator_cat/w1/Initializer/random_uniform/subSub3discriminator_cat/w1/Initializer/random_uniform/max3discriminator_cat/w1/Initializer/random_uniform/min*
T0*'
_class
loc:@discriminator_cat/w1
ą
3discriminator_cat/w1/Initializer/random_uniform/mulMul=discriminator_cat/w1/Initializer/random_uniform/RandomUniform3discriminator_cat/w1/Initializer/random_uniform/sub*
T0*'
_class
loc:@discriminator_cat/w1
Ņ
/discriminator_cat/w1/Initializer/random_uniformAdd3discriminator_cat/w1/Initializer/random_uniform/mul3discriminator_cat/w1/Initializer/random_uniform/min*
T0*'
_class
loc:@discriminator_cat/w1

discriminator_cat/w1
VariableV2*
shape:
*
shared_name *'
_class
loc:@discriminator_cat/w1*
dtype0*
	container 
Ē
discriminator_cat/w1/AssignAssigndiscriminator_cat/w1/discriminator_cat/w1/Initializer/random_uniform*
use_locking(*
T0*'
_class
loc:@discriminator_cat/w1*
validate_shape(
m
discriminator_cat/w1/readIdentitydiscriminator_cat/w1*
T0*'
_class
loc:@discriminator_cat/w1

&discriminator_cat/b1/Initializer/ConstConst*'
_class
loc:@discriminator_cat/b1*
valueB*    *
dtype0

discriminator_cat/b1
VariableV2*
shape:*
shared_name *'
_class
loc:@discriminator_cat/b1*
dtype0*
	container 
¾
discriminator_cat/b1/AssignAssigndiscriminator_cat/b1&discriminator_cat/b1/Initializer/Const*
use_locking(*
T0*'
_class
loc:@discriminator_cat/b1*
validate_shape(
m
discriminator_cat/b1/readIdentitydiscriminator_cat/b1*
T0*'
_class
loc:@discriminator_cat/b1

discriminator_cat/MatMul_1MatMuldiscriminator_cat/dropout/mul_1discriminator_cat/w1/read*
transpose_b( *
T0*
transpose_a( 
`
discriminator_cat/add_1AddV2discriminator_cat/MatMul_1discriminator_cat/b1/read*
T0
B
discriminator_cat/Relu_1Reludiscriminator_cat/add_1*
T0
F
discriminator_cat/sub_1/xConst*
valueB
 *  ?*
dtype0
M
discriminator_cat/sub_1Subdiscriminator_cat/sub_1/x	keep_prob*
T0
]
!discriminator_cat/dropout_1/ShapeShapediscriminator_cat/Relu_1*
T0*
out_type0
[
.discriminator_cat/dropout_1/random_uniform/minConst*
valueB
 *    *
dtype0
[
.discriminator_cat/dropout_1/random_uniform/maxConst*
valueB
 *  ?*
dtype0

8discriminator_cat/dropout_1/random_uniform/RandomUniformRandomUniform!discriminator_cat/dropout_1/Shape*

seed *
T0*
dtype0*
seed2 

.discriminator_cat/dropout_1/random_uniform/subSub.discriminator_cat/dropout_1/random_uniform/max.discriminator_cat/dropout_1/random_uniform/min*
T0
Ø
.discriminator_cat/dropout_1/random_uniform/mulMul8discriminator_cat/dropout_1/random_uniform/RandomUniform.discriminator_cat/dropout_1/random_uniform/sub*
T0

*discriminator_cat/dropout_1/random_uniformAdd.discriminator_cat/dropout_1/random_uniform/mul.discriminator_cat/dropout_1/random_uniform/min*
T0
N
!discriminator_cat/dropout_1/sub/xConst*
valueB
 *  ?*
dtype0
k
discriminator_cat/dropout_1/subSub!discriminator_cat/dropout_1/sub/xdiscriminator_cat/sub_1*
T0
R
%discriminator_cat/dropout_1/truediv/xConst*
valueB
 *  ?*
dtype0

#discriminator_cat/dropout_1/truedivRealDiv%discriminator_cat/dropout_1/truediv/xdiscriminator_cat/dropout_1/sub*
T0

(discriminator_cat/dropout_1/GreaterEqualGreaterEqual*discriminator_cat/dropout_1/random_uniformdiscriminator_cat/sub_1*
T0
n
discriminator_cat/dropout_1/mulMuldiscriminator_cat/Relu_1#discriminator_cat/dropout_1/truediv*
T0
z
 discriminator_cat/dropout_1/CastCast(discriminator_cat/dropout_1/GreaterEqual*

SrcT0
*
Truncate( *

DstT0
t
!discriminator_cat/dropout_1/mul_1Muldiscriminator_cat/dropout_1/mul discriminator_cat/dropout_1/Cast*
T0

5discriminator_cat/wo/Initializer/random_uniform/shapeConst*'
_class
loc:@discriminator_cat/wo*
valueB"      *
dtype0

3discriminator_cat/wo/Initializer/random_uniform/minConst*'
_class
loc:@discriminator_cat/wo*
valueB
 *Iv¾*
dtype0

3discriminator_cat/wo/Initializer/random_uniform/maxConst*'
_class
loc:@discriminator_cat/wo*
valueB
 *Iv>*
dtype0
Ż
=discriminator_cat/wo/Initializer/random_uniform/RandomUniformRandomUniform5discriminator_cat/wo/Initializer/random_uniform/shape*

seed *
T0*'
_class
loc:@discriminator_cat/wo*
dtype0*
seed2 
Ö
3discriminator_cat/wo/Initializer/random_uniform/subSub3discriminator_cat/wo/Initializer/random_uniform/max3discriminator_cat/wo/Initializer/random_uniform/min*
T0*'
_class
loc:@discriminator_cat/wo
ą
3discriminator_cat/wo/Initializer/random_uniform/mulMul=discriminator_cat/wo/Initializer/random_uniform/RandomUniform3discriminator_cat/wo/Initializer/random_uniform/sub*
T0*'
_class
loc:@discriminator_cat/wo
Ņ
/discriminator_cat/wo/Initializer/random_uniformAdd3discriminator_cat/wo/Initializer/random_uniform/mul3discriminator_cat/wo/Initializer/random_uniform/min*
T0*'
_class
loc:@discriminator_cat/wo

discriminator_cat/wo
VariableV2*
shape:	*
shared_name *'
_class
loc:@discriminator_cat/wo*
dtype0*
	container 
Ē
discriminator_cat/wo/AssignAssigndiscriminator_cat/wo/discriminator_cat/wo/Initializer/random_uniform*
use_locking(*
T0*'
_class
loc:@discriminator_cat/wo*
validate_shape(
m
discriminator_cat/wo/readIdentitydiscriminator_cat/wo*
T0*'
_class
loc:@discriminator_cat/wo

&discriminator_cat/bo/Initializer/ConstConst*'
_class
loc:@discriminator_cat/bo*
valueB*    *
dtype0

discriminator_cat/bo
VariableV2*
shape:*
shared_name *'
_class
loc:@discriminator_cat/bo*
dtype0*
	container 
¾
discriminator_cat/bo/AssignAssigndiscriminator_cat/bo&discriminator_cat/bo/Initializer/Const*
use_locking(*
T0*'
_class
loc:@discriminator_cat/bo*
validate_shape(
m
discriminator_cat/bo/readIdentitydiscriminator_cat/bo*
T0*'
_class
loc:@discriminator_cat/bo

discriminator_cat/MatMul_2MatMul!discriminator_cat/dropout_1/mul_1discriminator_cat/wo/read*
transpose_b( *
T0*
transpose_a( 
`
discriminator_cat/add_2AddV2discriminator_cat/MatMul_2discriminator_cat/bo/read*
T0
6
	Sigmoid_2Sigmoiddiscriminator_cat/add_2*
T0

discriminator_cat_1/MatMulMatMulCNN_encoder_cat/catout/Softmaxdiscriminator_cat/w0/read*
transpose_b( *
T0*
transpose_a( 
`
discriminator_cat_1/addAddV2discriminator_cat_1/MatMuldiscriminator_cat/b0/read*
T0
B
discriminator_cat_1/ReluReludiscriminator_cat_1/add*
T0
F
discriminator_cat_1/sub/xConst*
valueB
 *  ?*
dtype0
M
discriminator_cat_1/subSubdiscriminator_cat_1/sub/x	keep_prob*
T0
]
!discriminator_cat_1/dropout/ShapeShapediscriminator_cat_1/Relu*
T0*
out_type0
[
.discriminator_cat_1/dropout/random_uniform/minConst*
valueB
 *    *
dtype0
[
.discriminator_cat_1/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0

8discriminator_cat_1/dropout/random_uniform/RandomUniformRandomUniform!discriminator_cat_1/dropout/Shape*

seed *
T0*
dtype0*
seed2 

.discriminator_cat_1/dropout/random_uniform/subSub.discriminator_cat_1/dropout/random_uniform/max.discriminator_cat_1/dropout/random_uniform/min*
T0
Ø
.discriminator_cat_1/dropout/random_uniform/mulMul8discriminator_cat_1/dropout/random_uniform/RandomUniform.discriminator_cat_1/dropout/random_uniform/sub*
T0

*discriminator_cat_1/dropout/random_uniformAdd.discriminator_cat_1/dropout/random_uniform/mul.discriminator_cat_1/dropout/random_uniform/min*
T0
N
!discriminator_cat_1/dropout/sub/xConst*
valueB
 *  ?*
dtype0
k
discriminator_cat_1/dropout/subSub!discriminator_cat_1/dropout/sub/xdiscriminator_cat_1/sub*
T0
R
%discriminator_cat_1/dropout/truediv/xConst*
valueB
 *  ?*
dtype0

#discriminator_cat_1/dropout/truedivRealDiv%discriminator_cat_1/dropout/truediv/xdiscriminator_cat_1/dropout/sub*
T0

(discriminator_cat_1/dropout/GreaterEqualGreaterEqual*discriminator_cat_1/dropout/random_uniformdiscriminator_cat_1/sub*
T0
n
discriminator_cat_1/dropout/mulMuldiscriminator_cat_1/Relu#discriminator_cat_1/dropout/truediv*
T0
z
 discriminator_cat_1/dropout/CastCast(discriminator_cat_1/dropout/GreaterEqual*

SrcT0
*
Truncate( *

DstT0
t
!discriminator_cat_1/dropout/mul_1Muldiscriminator_cat_1/dropout/mul discriminator_cat_1/dropout/Cast*
T0

discriminator_cat_1/MatMul_1MatMul!discriminator_cat_1/dropout/mul_1discriminator_cat/w1/read*
transpose_b( *
T0*
transpose_a( 
d
discriminator_cat_1/add_1AddV2discriminator_cat_1/MatMul_1discriminator_cat/b1/read*
T0
F
discriminator_cat_1/Relu_1Reludiscriminator_cat_1/add_1*
T0
H
discriminator_cat_1/sub_1/xConst*
valueB
 *  ?*
dtype0
Q
discriminator_cat_1/sub_1Subdiscriminator_cat_1/sub_1/x	keep_prob*
T0
a
#discriminator_cat_1/dropout_1/ShapeShapediscriminator_cat_1/Relu_1*
T0*
out_type0
]
0discriminator_cat_1/dropout_1/random_uniform/minConst*
valueB
 *    *
dtype0
]
0discriminator_cat_1/dropout_1/random_uniform/maxConst*
valueB
 *  ?*
dtype0

:discriminator_cat_1/dropout_1/random_uniform/RandomUniformRandomUniform#discriminator_cat_1/dropout_1/Shape*

seed *
T0*
dtype0*
seed2 
¤
0discriminator_cat_1/dropout_1/random_uniform/subSub0discriminator_cat_1/dropout_1/random_uniform/max0discriminator_cat_1/dropout_1/random_uniform/min*
T0
®
0discriminator_cat_1/dropout_1/random_uniform/mulMul:discriminator_cat_1/dropout_1/random_uniform/RandomUniform0discriminator_cat_1/dropout_1/random_uniform/sub*
T0
 
,discriminator_cat_1/dropout_1/random_uniformAdd0discriminator_cat_1/dropout_1/random_uniform/mul0discriminator_cat_1/dropout_1/random_uniform/min*
T0
P
#discriminator_cat_1/dropout_1/sub/xConst*
valueB
 *  ?*
dtype0
q
!discriminator_cat_1/dropout_1/subSub#discriminator_cat_1/dropout_1/sub/xdiscriminator_cat_1/sub_1*
T0
T
'discriminator_cat_1/dropout_1/truediv/xConst*
valueB
 *  ?*
dtype0

%discriminator_cat_1/dropout_1/truedivRealDiv'discriminator_cat_1/dropout_1/truediv/x!discriminator_cat_1/dropout_1/sub*
T0

*discriminator_cat_1/dropout_1/GreaterEqualGreaterEqual,discriminator_cat_1/dropout_1/random_uniformdiscriminator_cat_1/sub_1*
T0
t
!discriminator_cat_1/dropout_1/mulMuldiscriminator_cat_1/Relu_1%discriminator_cat_1/dropout_1/truediv*
T0
~
"discriminator_cat_1/dropout_1/CastCast*discriminator_cat_1/dropout_1/GreaterEqual*

SrcT0
*
Truncate( *

DstT0
z
#discriminator_cat_1/dropout_1/mul_1Mul!discriminator_cat_1/dropout_1/mul"discriminator_cat_1/dropout_1/Cast*
T0

discriminator_cat_1/MatMul_2MatMul#discriminator_cat_1/dropout_1/mul_1discriminator_cat/wo/read*
transpose_b( *
T0*
transpose_a( 
d
discriminator_cat_1/add_2AddV2discriminator_cat_1/MatMul_2discriminator_cat/bo/read*
T0
8
	Sigmoid_3Sigmoiddiscriminator_cat_1/add_2*
T0
L
ones_like_2/ShapeShapediscriminator_cat/add_2*
T0*
out_type0
>
ones_like_2/ConstConst*
valueB
 *  ?*
dtype0
T
ones_like_2Fillones_like_2/Shapeones_like_2/Const*
T0*

index_type0
I
logistic_loss_3/zeros_like	ZerosLikediscriminator_cat/add_2*
T0
j
logistic_loss_3/GreaterEqualGreaterEqualdiscriminator_cat/add_2logistic_loss_3/zeros_like*
T0
|
logistic_loss_3/SelectSelectlogistic_loss_3/GreaterEqualdiscriminator_cat/add_2logistic_loss_3/zeros_like*
T0
<
logistic_loss_3/NegNegdiscriminator_cat/add_2*
T0
w
logistic_loss_3/Select_1Selectlogistic_loss_3/GreaterEquallogistic_loss_3/Negdiscriminator_cat/add_2*
T0
I
logistic_loss_3/mulMuldiscriminator_cat/add_2ones_like_2*
T0
P
logistic_loss_3/subSublogistic_loss_3/Selectlogistic_loss_3/mul*
T0
=
logistic_loss_3/ExpExplogistic_loss_3/Select_1*
T0
<
logistic_loss_3/Log1pLog1plogistic_loss_3/Exp*
T0
K
logistic_loss_3Addlogistic_loss_3/sublogistic_loss_3/Log1p*
T0
<
Const_4Const*
valueB"       *
dtype0
N
Mean_4Meanlogistic_loss_3Const_4*

Tidx0*
	keep_dims( *
T0
=
zeros_like_1	ZerosLikediscriminator_cat_1/add_2*
T0
K
logistic_loss_4/zeros_like	ZerosLikediscriminator_cat_1/add_2*
T0
l
logistic_loss_4/GreaterEqualGreaterEqualdiscriminator_cat_1/add_2logistic_loss_4/zeros_like*
T0
~
logistic_loss_4/SelectSelectlogistic_loss_4/GreaterEqualdiscriminator_cat_1/add_2logistic_loss_4/zeros_like*
T0
>
logistic_loss_4/NegNegdiscriminator_cat_1/add_2*
T0
y
logistic_loss_4/Select_1Selectlogistic_loss_4/GreaterEquallogistic_loss_4/Negdiscriminator_cat_1/add_2*
T0
L
logistic_loss_4/mulMuldiscriminator_cat_1/add_2zeros_like_1*
T0
P
logistic_loss_4/subSublogistic_loss_4/Selectlogistic_loss_4/mul*
T0
=
logistic_loss_4/ExpExplogistic_loss_4/Select_1*
T0
<
logistic_loss_4/Log1pLog1plogistic_loss_4/Exp*
T0
K
logistic_loss_4Addlogistic_loss_4/sublogistic_loss_4/Log1p*
T0
<
Const_5Const*
valueB"       *
dtype0
N
Mean_5Meanlogistic_loss_4Const_5*

Tidx0*
	keep_dims( *
T0
'
add_1AddV2Mean_4Mean_5*
T0
N
ones_like_3/ShapeShapediscriminator_cat_1/add_2*
T0*
out_type0
>
ones_like_3/ConstConst*
valueB
 *  ?*
dtype0
T
ones_like_3Fillones_like_3/Shapeones_like_3/Const*
T0*

index_type0
K
logistic_loss_5/zeros_like	ZerosLikediscriminator_cat_1/add_2*
T0
l
logistic_loss_5/GreaterEqualGreaterEqualdiscriminator_cat_1/add_2logistic_loss_5/zeros_like*
T0
~
logistic_loss_5/SelectSelectlogistic_loss_5/GreaterEqualdiscriminator_cat_1/add_2logistic_loss_5/zeros_like*
T0
>
logistic_loss_5/NegNegdiscriminator_cat_1/add_2*
T0
y
logistic_loss_5/Select_1Selectlogistic_loss_5/GreaterEquallogistic_loss_5/Negdiscriminator_cat_1/add_2*
T0
K
logistic_loss_5/mulMuldiscriminator_cat_1/add_2ones_like_3*
T0
P
logistic_loss_5/subSublogistic_loss_5/Selectlogistic_loss_5/mul*
T0
=
logistic_loss_5/ExpExplogistic_loss_5/Select_1*
T0
<
logistic_loss_5/Log1pLog1plogistic_loss_5/Exp*
T0
K
logistic_loss_5Addlogistic_loss_5/sublogistic_loss_5/Log1p*
T0
<
Const_6Const*
valueB"       *
dtype0
N
Mean_6Meanlogistic_loss_5Const_6*

Tidx0*
	keep_dims( *
T0
#
add_2AddV2addadd_1*
T0
0
Const_7Const*
valueB *
dtype0
D
Mean_7Meanadd_2Const_7*

Tidx0*
	keep_dims( *
T0
'
add_3AddV2Mean_3Mean_6*
T0
0
Const_8Const*
valueB *
dtype0
D
Mean_8Meanadd_3Const_8*

Tidx0*
	keep_dims( *
T0
c
9softmax_cross_entropy_with_logits_sg/labels_stop_gradientStopGradientinput_img_label*
T0
S
)softmax_cross_entropy_with_logits_sg/RankConst*
value	B :*
dtype0
l
*softmax_cross_entropy_with_logits_sg/ShapeShapeCNN_encoder_cat/catout/Softmax*
T0*
out_type0
U
+softmax_cross_entropy_with_logits_sg/Rank_1Const*
value	B :*
dtype0
n
,softmax_cross_entropy_with_logits_sg/Shape_1ShapeCNN_encoder_cat/catout/Softmax*
T0*
out_type0
T
*softmax_cross_entropy_with_logits_sg/Sub/yConst*
value	B :*
dtype0

(softmax_cross_entropy_with_logits_sg/SubSub+softmax_cross_entropy_with_logits_sg/Rank_1*softmax_cross_entropy_with_logits_sg/Sub/y*
T0

0softmax_cross_entropy_with_logits_sg/Slice/beginPack(softmax_cross_entropy_with_logits_sg/Sub*
T0*

axis *
N
]
/softmax_cross_entropy_with_logits_sg/Slice/sizeConst*
valueB:*
dtype0
Ś
*softmax_cross_entropy_with_logits_sg/SliceSlice,softmax_cross_entropy_with_logits_sg/Shape_10softmax_cross_entropy_with_logits_sg/Slice/begin/softmax_cross_entropy_with_logits_sg/Slice/size*
T0*
Index0
k
4softmax_cross_entropy_with_logits_sg/concat/values_0Const*
valueB:
’’’’’’’’’*
dtype0
Z
0softmax_cross_entropy_with_logits_sg/concat/axisConst*
value	B : *
dtype0
é
+softmax_cross_entropy_with_logits_sg/concatConcatV24softmax_cross_entropy_with_logits_sg/concat/values_0*softmax_cross_entropy_with_logits_sg/Slice0softmax_cross_entropy_with_logits_sg/concat/axis*

Tidx0*
T0*
N

,softmax_cross_entropy_with_logits_sg/ReshapeReshapeCNN_encoder_cat/catout/Softmax+softmax_cross_entropy_with_logits_sg/concat*
T0*
Tshape0
U
+softmax_cross_entropy_with_logits_sg/Rank_2Const*
value	B :*
dtype0

,softmax_cross_entropy_with_logits_sg/Shape_2Shape9softmax_cross_entropy_with_logits_sg/labels_stop_gradient*
T0*
out_type0
V
,softmax_cross_entropy_with_logits_sg/Sub_1/yConst*
value	B :*
dtype0

*softmax_cross_entropy_with_logits_sg/Sub_1Sub+softmax_cross_entropy_with_logits_sg/Rank_2,softmax_cross_entropy_with_logits_sg/Sub_1/y*
T0

2softmax_cross_entropy_with_logits_sg/Slice_1/beginPack*softmax_cross_entropy_with_logits_sg/Sub_1*
T0*

axis *
N
_
1softmax_cross_entropy_with_logits_sg/Slice_1/sizeConst*
valueB:*
dtype0
ą
,softmax_cross_entropy_with_logits_sg/Slice_1Slice,softmax_cross_entropy_with_logits_sg/Shape_22softmax_cross_entropy_with_logits_sg/Slice_1/begin1softmax_cross_entropy_with_logits_sg/Slice_1/size*
T0*
Index0
m
6softmax_cross_entropy_with_logits_sg/concat_1/values_0Const*
valueB:
’’’’’’’’’*
dtype0
\
2softmax_cross_entropy_with_logits_sg/concat_1/axisConst*
value	B : *
dtype0
ń
-softmax_cross_entropy_with_logits_sg/concat_1ConcatV26softmax_cross_entropy_with_logits_sg/concat_1/values_0,softmax_cross_entropy_with_logits_sg/Slice_12softmax_cross_entropy_with_logits_sg/concat_1/axis*

Tidx0*
T0*
N
ŗ
.softmax_cross_entropy_with_logits_sg/Reshape_1Reshape9softmax_cross_entropy_with_logits_sg/labels_stop_gradient-softmax_cross_entropy_with_logits_sg/concat_1*
T0*
Tshape0
¬
$softmax_cross_entropy_with_logits_sgSoftmaxCrossEntropyWithLogits,softmax_cross_entropy_with_logits_sg/Reshape.softmax_cross_entropy_with_logits_sg/Reshape_1*
T0
V
,softmax_cross_entropy_with_logits_sg/Sub_2/yConst*
value	B :*
dtype0

*softmax_cross_entropy_with_logits_sg/Sub_2Sub)softmax_cross_entropy_with_logits_sg/Rank,softmax_cross_entropy_with_logits_sg/Sub_2/y*
T0
`
2softmax_cross_entropy_with_logits_sg/Slice_2/beginConst*
valueB: *
dtype0

1softmax_cross_entropy_with_logits_sg/Slice_2/sizePack*softmax_cross_entropy_with_logits_sg/Sub_2*
T0*

axis *
N
Ž
,softmax_cross_entropy_with_logits_sg/Slice_2Slice*softmax_cross_entropy_with_logits_sg/Shape2softmax_cross_entropy_with_logits_sg/Slice_2/begin1softmax_cross_entropy_with_logits_sg/Slice_2/size*
T0*
Index0
¤
.softmax_cross_entropy_with_logits_sg/Reshape_2Reshape$softmax_cross_entropy_with_logits_sg,softmax_cross_entropy_with_logits_sg/Slice_2*
T0*
Tshape0
5
Const_9Const*
valueB: *
dtype0
m
Mean_9Mean.softmax_cross_entropy_with_logits_sg/Reshape_2Const_9*

Tidx0*
	keep_dims( *
T0
8
gradients/ShapeConst*
valueB *
dtype0
@
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0
W
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0
O
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0
p
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0
@
gradients/Mean_grad/ShapeShapeSum*
T0*
out_type0
s
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Shape*

Tmultiples0*
T0
B
gradients/Mean_grad/Shape_1ShapeSum*
T0*
out_type0
D
gradients/Mean_grad/Shape_2Const*
valueB *
dtype0
G
gradients/Mean_grad/ConstConst*
valueB: *
dtype0
~
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_1gradients/Mean_grad/Const*

Tidx0*
	keep_dims( *
T0
I
gradients/Mean_grad/Const_1Const*
valueB: *
dtype0

gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_2gradients/Mean_grad/Const_1*

Tidx0*
	keep_dims( *
T0
G
gradients/Mean_grad/Maximum/yConst*
value	B :*
dtype0
j
gradients/Mean_grad/MaximumMaximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum/y*
T0
h
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum*
T0
f
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv*

SrcT0*
Truncate( *

DstT0
c
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
T0
M
gradients/Sum_grad/ShapeShapeSquaredDifference*
T0*
out_type0
n
gradients/Sum_grad/SizeConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0

gradients/Sum_grad/addAddV2Sum/reduction_indicesgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape

gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape
u
gradients/Sum_grad/Shape_1Const*+
_class!
loc:@gradients/Sum_grad/Shape*
valueB:*
dtype0
u
gradients/Sum_grad/range/startConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B : *
dtype0
u
gradients/Sum_grad/range/deltaConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0
³
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*

Tidx0*+
_class!
loc:@gradients/Sum_grad/Shape
t
gradients/Sum_grad/Fill/valueConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0
¢
gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*

index_type0
Õ
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
N
s
gradients/Sum_grad/Maximum/yConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0

gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
T0*+
_class!
loc:@gradients/Sum_grad/Shape

gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
T0*+
_class!
loc:@gradients/Sum_grad/Shape
{
gradients/Sum_grad/ReshapeReshapegradients/Mean_grad/truediv gradients/Sum_grad/DynamicStitch*
T0*
Tshape0
s
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*

Tmultiples0*
T0
n
'gradients/SquaredDifference_grad/scalarConst^gradients/Sum_grad/Tile*
valueB
 *   @*
dtype0
v
$gradients/SquaredDifference_grad/MulMul'gradients/SquaredDifference_grad/scalargradients/Sum_grad/Tile*
T0
x
$gradients/SquaredDifference_grad/subSub
target_imgCNN_decoder/reshaped/Reshape^gradients/Sum_grad/Tile*
T0

&gradients/SquaredDifference_grad/mul_1Mul$gradients/SquaredDifference_grad/Mul$gradients/SquaredDifference_grad/sub*
T0
T
&gradients/SquaredDifference_grad/ShapeShape
target_img*
T0*
out_type0
h
(gradients/SquaredDifference_grad/Shape_1ShapeCNN_decoder/reshaped/Reshape*
T0*
out_type0
Ŗ
6gradients/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/SquaredDifference_grad/Shape(gradients/SquaredDifference_grad/Shape_1*
T0
±
$gradients/SquaredDifference_grad/SumSum&gradients/SquaredDifference_grad/mul_16gradients/SquaredDifference_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0

(gradients/SquaredDifference_grad/ReshapeReshape$gradients/SquaredDifference_grad/Sum&gradients/SquaredDifference_grad/Shape*
T0*
Tshape0
µ
&gradients/SquaredDifference_grad/Sum_1Sum&gradients/SquaredDifference_grad/mul_18gradients/SquaredDifference_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0

*gradients/SquaredDifference_grad/Reshape_1Reshape&gradients/SquaredDifference_grad/Sum_1(gradients/SquaredDifference_grad/Shape_1*
T0*
Tshape0
`
$gradients/SquaredDifference_grad/NegNeg*gradients/SquaredDifference_grad/Reshape_1*
T0

1gradients/SquaredDifference_grad/tuple/group_depsNoOp%^gradients/SquaredDifference_grad/Neg)^gradients/SquaredDifference_grad/Reshape
é
9gradients/SquaredDifference_grad/tuple/control_dependencyIdentity(gradients/SquaredDifference_grad/Reshape2^gradients/SquaredDifference_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/SquaredDifference_grad/Reshape
ć
;gradients/SquaredDifference_grad/tuple/control_dependency_1Identity$gradients/SquaredDifference_grad/Neg2^gradients/SquaredDifference_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/SquaredDifference_grad/Neg
q
1gradients/CNN_decoder/reshaped/Reshape_grad/ShapeShapeCNN_decoder/sigout_1/BiasAdd*
T0*
out_type0
Å
3gradients/CNN_decoder/reshaped/Reshape_grad/ReshapeReshape;gradients/SquaredDifference_grad/tuple/control_dependency_11gradients/CNN_decoder/reshaped/Reshape_grad/Shape*
T0*
Tshape0

7gradients/CNN_decoder/sigout_1/BiasAdd_grad/BiasAddGradBiasAddGrad3gradients/CNN_decoder/reshaped/Reshape_grad/Reshape*
T0*
data_formatNHWC
“
<gradients/CNN_decoder/sigout_1/BiasAdd_grad/tuple/group_depsNoOp4^gradients/CNN_decoder/reshaped/Reshape_grad/Reshape8^gradients/CNN_decoder/sigout_1/BiasAdd_grad/BiasAddGrad

Dgradients/CNN_decoder/sigout_1/BiasAdd_grad/tuple/control_dependencyIdentity3gradients/CNN_decoder/reshaped/Reshape_grad/Reshape=^gradients/CNN_decoder/sigout_1/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/CNN_decoder/reshaped/Reshape_grad/Reshape

Fgradients/CNN_decoder/sigout_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/CNN_decoder/sigout_1/BiasAdd_grad/BiasAddGrad=^gradients/CNN_decoder/sigout_1/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/CNN_decoder/sigout_1/BiasAdd_grad/BiasAddGrad
Ķ
1gradients/CNN_decoder/sigout_1/MatMul_grad/MatMulMatMulDgradients/CNN_decoder/sigout_1/BiasAdd_grad/tuple/control_dependencyCNN_decoder/sigout_1/W/read*
transpose_b(*
T0*
transpose_a( 
Š
3gradients/CNN_decoder/sigout_1/MatMul_grad/MatMul_1MatMulCNN_decoder/sigout_1/ReshapeDgradients/CNN_decoder/sigout_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
­
;gradients/CNN_decoder/sigout_1/MatMul_grad/tuple/group_depsNoOp2^gradients/CNN_decoder/sigout_1/MatMul_grad/MatMul4^gradients/CNN_decoder/sigout_1/MatMul_grad/MatMul_1

Cgradients/CNN_decoder/sigout_1/MatMul_grad/tuple/control_dependencyIdentity1gradients/CNN_decoder/sigout_1/MatMul_grad/MatMul<^gradients/CNN_decoder/sigout_1/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/CNN_decoder/sigout_1/MatMul_grad/MatMul

Egradients/CNN_decoder/sigout_1/MatMul_grad/tuple/control_dependency_1Identity3gradients/CNN_decoder/sigout_1/MatMul_grad/MatMul_1<^gradients/CNN_decoder/sigout_1/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/CNN_decoder/sigout_1/MatMul_grad/MatMul_1
l
1gradients/CNN_decoder/sigout_1/Reshape_grad/ShapeShapeCNN_decoder/sigout/Tanh*
T0*
out_type0
Ķ
3gradients/CNN_decoder/sigout_1/Reshape_grad/ReshapeReshapeCgradients/CNN_decoder/sigout_1/MatMul_grad/tuple/control_dependency1gradients/CNN_decoder/sigout_1/Reshape_grad/Shape*
T0*
Tshape0

/gradients/CNN_decoder/sigout/Tanh_grad/TanhGradTanhGradCNN_decoder/sigout/Tanh3gradients/CNN_decoder/sigout_1/Reshape_grad/Reshape*
T0

5gradients/CNN_decoder/sigout/BiasAdd_grad/BiasAddGradBiasAddGrad/gradients/CNN_decoder/sigout/Tanh_grad/TanhGrad*
T0*
data_formatNHWC
¬
:gradients/CNN_decoder/sigout/BiasAdd_grad/tuple/group_depsNoOp6^gradients/CNN_decoder/sigout/BiasAdd_grad/BiasAddGrad0^gradients/CNN_decoder/sigout/Tanh_grad/TanhGrad

Bgradients/CNN_decoder/sigout/BiasAdd_grad/tuple/control_dependencyIdentity/gradients/CNN_decoder/sigout/Tanh_grad/TanhGrad;^gradients/CNN_decoder/sigout/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/CNN_decoder/sigout/Tanh_grad/TanhGrad

Dgradients/CNN_decoder/sigout/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/CNN_decoder/sigout/BiasAdd_grad/BiasAddGrad;^gradients/CNN_decoder/sigout/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/CNN_decoder/sigout/BiasAdd_grad/BiasAddGrad
¦
/gradients/CNN_decoder/sigout/Conv2D_grad/ShapeNShapeN.CNN_decoder/UpSample2D_2/ResizeNearestNeighborCNN_decoder/sigout/W/read*
T0*
out_type0*
N
ó
<gradients/CNN_decoder/sigout/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput/gradients/CNN_decoder/sigout/Conv2D_grad/ShapeNCNN_decoder/sigout/W/readBgradients/CNN_decoder/sigout/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME

=gradients/CNN_decoder/sigout/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter.CNN_decoder/UpSample2D_2/ResizeNearestNeighbor1gradients/CNN_decoder/sigout/Conv2D_grad/ShapeN:1Bgradients/CNN_decoder/sigout/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
Ą
9gradients/CNN_decoder/sigout/Conv2D_grad/tuple/group_depsNoOp>^gradients/CNN_decoder/sigout/Conv2D_grad/Conv2DBackpropFilter=^gradients/CNN_decoder/sigout/Conv2D_grad/Conv2DBackpropInput
”
Agradients/CNN_decoder/sigout/Conv2D_grad/tuple/control_dependencyIdentity<gradients/CNN_decoder/sigout/Conv2D_grad/Conv2DBackpropInput:^gradients/CNN_decoder/sigout/Conv2D_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/CNN_decoder/sigout/Conv2D_grad/Conv2DBackpropInput
„
Cgradients/CNN_decoder/sigout/Conv2D_grad/tuple/control_dependency_1Identity=gradients/CNN_decoder/sigout/Conv2D_grad/Conv2DBackpropFilter:^gradients/CNN_decoder/sigout/Conv2D_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/CNN_decoder/sigout/Conv2D_grad/Conv2DBackpropFilter

\gradients/CNN_decoder/UpSample2D_2/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*
valueB"      *
dtype0
Ķ
Wgradients/CNN_decoder/UpSample2D_2/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradAgradients/CNN_decoder/sigout/Conv2D_grad/tuple/control_dependency\gradients/CNN_decoder/UpSample2D_2/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*
align_corners( *
half_pixel_centers( *
T0
ŗ
1gradients/CNN_decoder/Conv2D_2/Tanh_grad/TanhGradTanhGradCNN_decoder/Conv2D_2/TanhWgradients/CNN_decoder/UpSample2D_2/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad*
T0

7gradients/CNN_decoder/Conv2D_2/BiasAdd_grad/BiasAddGradBiasAddGrad1gradients/CNN_decoder/Conv2D_2/Tanh_grad/TanhGrad*
T0*
data_formatNHWC
²
<gradients/CNN_decoder/Conv2D_2/BiasAdd_grad/tuple/group_depsNoOp8^gradients/CNN_decoder/Conv2D_2/BiasAdd_grad/BiasAddGrad2^gradients/CNN_decoder/Conv2D_2/Tanh_grad/TanhGrad

Dgradients/CNN_decoder/Conv2D_2/BiasAdd_grad/tuple/control_dependencyIdentity1gradients/CNN_decoder/Conv2D_2/Tanh_grad/TanhGrad=^gradients/CNN_decoder/Conv2D_2/BiasAdd_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/CNN_decoder/Conv2D_2/Tanh_grad/TanhGrad

Fgradients/CNN_decoder/Conv2D_2/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/CNN_decoder/Conv2D_2/BiasAdd_grad/BiasAddGrad=^gradients/CNN_decoder/Conv2D_2/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/CNN_decoder/Conv2D_2/BiasAdd_grad/BiasAddGrad
Ŗ
1gradients/CNN_decoder/Conv2D_2/Conv2D_grad/ShapeNShapeN.CNN_decoder/UpSample2D_1/ResizeNearestNeighborCNN_decoder/Conv2D_2/W/read*
T0*
out_type0*
N
ū
>gradients/CNN_decoder/Conv2D_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput1gradients/CNN_decoder/Conv2D_2/Conv2D_grad/ShapeNCNN_decoder/Conv2D_2/W/readDgradients/CNN_decoder/Conv2D_2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME

?gradients/CNN_decoder/Conv2D_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter.CNN_decoder/UpSample2D_1/ResizeNearestNeighbor3gradients/CNN_decoder/Conv2D_2/Conv2D_grad/ShapeN:1Dgradients/CNN_decoder/Conv2D_2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
Ę
;gradients/CNN_decoder/Conv2D_2/Conv2D_grad/tuple/group_depsNoOp@^gradients/CNN_decoder/Conv2D_2/Conv2D_grad/Conv2DBackpropFilter?^gradients/CNN_decoder/Conv2D_2/Conv2D_grad/Conv2DBackpropInput
©
Cgradients/CNN_decoder/Conv2D_2/Conv2D_grad/tuple/control_dependencyIdentity>gradients/CNN_decoder/Conv2D_2/Conv2D_grad/Conv2DBackpropInput<^gradients/CNN_decoder/Conv2D_2/Conv2D_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/CNN_decoder/Conv2D_2/Conv2D_grad/Conv2DBackpropInput
­
Egradients/CNN_decoder/Conv2D_2/Conv2D_grad/tuple/control_dependency_1Identity?gradients/CNN_decoder/Conv2D_2/Conv2D_grad/Conv2DBackpropFilter<^gradients/CNN_decoder/Conv2D_2/Conv2D_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/CNN_decoder/Conv2D_2/Conv2D_grad/Conv2DBackpropFilter

\gradients/CNN_decoder/UpSample2D_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*
valueB"   L   *
dtype0
Ļ
Wgradients/CNN_decoder/UpSample2D_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradCgradients/CNN_decoder/Conv2D_2/Conv2D_grad/tuple/control_dependency\gradients/CNN_decoder/UpSample2D_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*
align_corners( *
half_pixel_centers( *
T0
ŗ
1gradients/CNN_decoder/Conv2D_1/Tanh_grad/TanhGradTanhGradCNN_decoder/Conv2D_1/TanhWgradients/CNN_decoder/UpSample2D_1/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad*
T0

7gradients/CNN_decoder/Conv2D_1/BiasAdd_grad/BiasAddGradBiasAddGrad1gradients/CNN_decoder/Conv2D_1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC
²
<gradients/CNN_decoder/Conv2D_1/BiasAdd_grad/tuple/group_depsNoOp8^gradients/CNN_decoder/Conv2D_1/BiasAdd_grad/BiasAddGrad2^gradients/CNN_decoder/Conv2D_1/Tanh_grad/TanhGrad

Dgradients/CNN_decoder/Conv2D_1/BiasAdd_grad/tuple/control_dependencyIdentity1gradients/CNN_decoder/Conv2D_1/Tanh_grad/TanhGrad=^gradients/CNN_decoder/Conv2D_1/BiasAdd_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/CNN_decoder/Conv2D_1/Tanh_grad/TanhGrad

Fgradients/CNN_decoder/Conv2D_1/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/CNN_decoder/Conv2D_1/BiasAdd_grad/BiasAddGrad=^gradients/CNN_decoder/Conv2D_1/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/CNN_decoder/Conv2D_1/BiasAdd_grad/BiasAddGrad
Ø
1gradients/CNN_decoder/Conv2D_1/Conv2D_grad/ShapeNShapeN,CNN_decoder/UpSample2D/ResizeNearestNeighborCNN_decoder/Conv2D_1/W/read*
T0*
out_type0*
N
ū
>gradients/CNN_decoder/Conv2D_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput1gradients/CNN_decoder/Conv2D_1/Conv2D_grad/ShapeNCNN_decoder/Conv2D_1/W/readDgradients/CNN_decoder/Conv2D_1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME

?gradients/CNN_decoder/Conv2D_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter,CNN_decoder/UpSample2D/ResizeNearestNeighbor3gradients/CNN_decoder/Conv2D_1/Conv2D_grad/ShapeN:1Dgradients/CNN_decoder/Conv2D_1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
Ę
;gradients/CNN_decoder/Conv2D_1/Conv2D_grad/tuple/group_depsNoOp@^gradients/CNN_decoder/Conv2D_1/Conv2D_grad/Conv2DBackpropFilter?^gradients/CNN_decoder/Conv2D_1/Conv2D_grad/Conv2DBackpropInput
©
Cgradients/CNN_decoder/Conv2D_1/Conv2D_grad/tuple/control_dependencyIdentity>gradients/CNN_decoder/Conv2D_1/Conv2D_grad/Conv2DBackpropInput<^gradients/CNN_decoder/Conv2D_1/Conv2D_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/CNN_decoder/Conv2D_1/Conv2D_grad/Conv2DBackpropInput
­
Egradients/CNN_decoder/Conv2D_1/Conv2D_grad/tuple/control_dependency_1Identity?gradients/CNN_decoder/Conv2D_1/Conv2D_grad/Conv2DBackpropFilter<^gradients/CNN_decoder/Conv2D_1/Conv2D_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/CNN_decoder/Conv2D_1/Conv2D_grad/Conv2DBackpropFilter

Zgradients/CNN_decoder/UpSample2D/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/sizeConst*
valueB"   &   *
dtype0
Ė
Ugradients/CNN_decoder/UpSample2D/ResizeNearestNeighbor_grad/ResizeNearestNeighborGradResizeNearestNeighborGradCgradients/CNN_decoder/Conv2D_1/Conv2D_grad/tuple/control_dependencyZgradients/CNN_decoder/UpSample2D/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad/size*
align_corners( *
half_pixel_centers( *
T0
“
/gradients/CNN_decoder/Conv2D/Tanh_grad/TanhGradTanhGradCNN_decoder/Conv2D/TanhUgradients/CNN_decoder/UpSample2D/ResizeNearestNeighbor_grad/ResizeNearestNeighborGrad*
T0

5gradients/CNN_decoder/Conv2D/BiasAdd_grad/BiasAddGradBiasAddGrad/gradients/CNN_decoder/Conv2D/Tanh_grad/TanhGrad*
T0*
data_formatNHWC
¬
:gradients/CNN_decoder/Conv2D/BiasAdd_grad/tuple/group_depsNoOp6^gradients/CNN_decoder/Conv2D/BiasAdd_grad/BiasAddGrad0^gradients/CNN_decoder/Conv2D/Tanh_grad/TanhGrad

Bgradients/CNN_decoder/Conv2D/BiasAdd_grad/tuple/control_dependencyIdentity/gradients/CNN_decoder/Conv2D/Tanh_grad/TanhGrad;^gradients/CNN_decoder/Conv2D/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/CNN_decoder/Conv2D/Tanh_grad/TanhGrad

Dgradients/CNN_decoder/Conv2D/BiasAdd_grad/tuple/control_dependency_1Identity5gradients/CNN_decoder/Conv2D/BiasAdd_grad/BiasAddGrad;^gradients/CNN_decoder/Conv2D/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/CNN_decoder/Conv2D/BiasAdd_grad/BiasAddGrad

/gradients/CNN_decoder/Conv2D/Conv2D_grad/ShapeNShapeNCNN_decoder/ReshapeCNN_decoder/Conv2D/W/read*
T0*
out_type0*
N
ó
<gradients/CNN_decoder/Conv2D/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput/gradients/CNN_decoder/Conv2D/Conv2D_grad/ShapeNCNN_decoder/Conv2D/W/readBgradients/CNN_decoder/Conv2D/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
ń
=gradients/CNN_decoder/Conv2D/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterCNN_decoder/Reshape1gradients/CNN_decoder/Conv2D/Conv2D_grad/ShapeN:1Bgradients/CNN_decoder/Conv2D/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
Ą
9gradients/CNN_decoder/Conv2D/Conv2D_grad/tuple/group_depsNoOp>^gradients/CNN_decoder/Conv2D/Conv2D_grad/Conv2DBackpropFilter=^gradients/CNN_decoder/Conv2D/Conv2D_grad/Conv2DBackpropInput
”
Agradients/CNN_decoder/Conv2D/Conv2D_grad/tuple/control_dependencyIdentity<gradients/CNN_decoder/Conv2D/Conv2D_grad/Conv2DBackpropInput:^gradients/CNN_decoder/Conv2D/Conv2D_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/CNN_decoder/Conv2D/Conv2D_grad/Conv2DBackpropInput
„
Cgradients/CNN_decoder/Conv2D/Conv2D_grad/tuple/control_dependency_1Identity=gradients/CNN_decoder/Conv2D/Conv2D_grad/Conv2DBackpropFilter:^gradients/CNN_decoder/Conv2D/Conv2D_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/CNN_decoder/Conv2D/Conv2D_grad/Conv2DBackpropFilter
k
(gradients/CNN_decoder/Reshape_grad/ShapeShapeCNN_decoder/FullyConnected/Tanh*
T0*
out_type0
¹
*gradients/CNN_decoder/Reshape_grad/ReshapeReshapeAgradients/CNN_decoder/Conv2D/Conv2D_grad/tuple/control_dependency(gradients/CNN_decoder/Reshape_grad/Shape*
T0*
Tshape0

7gradients/CNN_decoder/FullyConnected/Tanh_grad/TanhGradTanhGradCNN_decoder/FullyConnected/Tanh*gradients/CNN_decoder/Reshape_grad/Reshape*
T0
„
=gradients/CNN_decoder/FullyConnected/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients/CNN_decoder/FullyConnected/Tanh_grad/TanhGrad*
T0*
data_formatNHWC
Ä
Bgradients/CNN_decoder/FullyConnected/BiasAdd_grad/tuple/group_depsNoOp>^gradients/CNN_decoder/FullyConnected/BiasAdd_grad/BiasAddGrad8^gradients/CNN_decoder/FullyConnected/Tanh_grad/TanhGrad
©
Jgradients/CNN_decoder/FullyConnected/BiasAdd_grad/tuple/control_dependencyIdentity7gradients/CNN_decoder/FullyConnected/Tanh_grad/TanhGradC^gradients/CNN_decoder/FullyConnected/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/CNN_decoder/FullyConnected/Tanh_grad/TanhGrad
·
Lgradients/CNN_decoder/FullyConnected/BiasAdd_grad/tuple/control_dependency_1Identity=gradients/CNN_decoder/FullyConnected/BiasAdd_grad/BiasAddGradC^gradients/CNN_decoder/FullyConnected/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/CNN_decoder/FullyConnected/BiasAdd_grad/BiasAddGrad
ß
7gradients/CNN_decoder/FullyConnected/MatMul_grad/MatMulMatMulJgradients/CNN_decoder/FullyConnected/BiasAdd_grad/tuple/control_dependency!CNN_decoder/FullyConnected/W/read*
transpose_b(*
T0*
transpose_a( 
Ę
9gradients/CNN_decoder/FullyConnected/MatMul_grad/MatMul_1MatMulconcatJgradients/CNN_decoder/FullyConnected/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
æ
Agradients/CNN_decoder/FullyConnected/MatMul_grad/tuple/group_depsNoOp8^gradients/CNN_decoder/FullyConnected/MatMul_grad/MatMul:^gradients/CNN_decoder/FullyConnected/MatMul_grad/MatMul_1
§
Igradients/CNN_decoder/FullyConnected/MatMul_grad/tuple/control_dependencyIdentity7gradients/CNN_decoder/FullyConnected/MatMul_grad/MatMulB^gradients/CNN_decoder/FullyConnected/MatMul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/CNN_decoder/FullyConnected/MatMul_grad/MatMul
­
Kgradients/CNN_decoder/FullyConnected/MatMul_grad/tuple/control_dependency_1Identity9gradients/CNN_decoder/FullyConnected/MatMul_grad/MatMul_1B^gradients/CNN_decoder/FullyConnected/MatMul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/CNN_decoder/FullyConnected/MatMul_grad/MatMul_1
D
gradients/concat_grad/RankConst*
value	B :*
dtype0
W
gradients/concat_grad/modFloorModconcat/axisgradients/concat_grad/Rank*
T0
[
gradients/concat_grad/ShapeShapeCNN_encoder_cat/zout/BiasAdd*
T0*
out_type0

gradients/concat_grad/ShapeNShapeNCNN_encoder_cat/zout/BiasAddCNN_encoder_cat/catout/Softmax*
T0*
out_type0*
N

"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/modgradients/concat_grad/ShapeNgradients/concat_grad/ShapeN:1*
N
Ē
gradients/concat_grad/SliceSliceIgradients/CNN_decoder/FullyConnected/MatMul_grad/tuple/control_dependency"gradients/concat_grad/ConcatOffsetgradients/concat_grad/ShapeN*
T0*
Index0
Ķ
gradients/concat_grad/Slice_1SliceIgradients/CNN_decoder/FullyConnected/MatMul_grad/tuple/control_dependency$gradients/concat_grad/ConcatOffset:1gradients/concat_grad/ShapeN:1*
T0*
Index0
l
&gradients/concat_grad/tuple/group_depsNoOp^gradients/concat_grad/Slice^gradients/concat_grad/Slice_1
¹
.gradients/concat_grad/tuple/control_dependencyIdentitygradients/concat_grad/Slice'^gradients/concat_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/concat_grad/Slice
æ
0gradients/concat_grad/tuple/control_dependency_1Identitygradients/concat_grad/Slice_1'^gradients/concat_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_grad/Slice_1

7gradients/CNN_encoder_cat/zout/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients/concat_grad/tuple/control_dependency*
T0*
data_formatNHWC
Æ
<gradients/CNN_encoder_cat/zout/BiasAdd_grad/tuple/group_depsNoOp8^gradients/CNN_encoder_cat/zout/BiasAdd_grad/BiasAddGrad/^gradients/concat_grad/tuple/control_dependency
ų
Dgradients/CNN_encoder_cat/zout/BiasAdd_grad/tuple/control_dependencyIdentity.gradients/concat_grad/tuple/control_dependency=^gradients/CNN_encoder_cat/zout/BiasAdd_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients/concat_grad/Slice

Fgradients/CNN_encoder_cat/zout/BiasAdd_grad/tuple/control_dependency_1Identity7gradients/CNN_encoder_cat/zout/BiasAdd_grad/BiasAddGrad=^gradients/CNN_encoder_cat/zout/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/CNN_encoder_cat/zout/BiasAdd_grad/BiasAddGrad

1gradients/CNN_encoder_cat/catout/Softmax_grad/mulMul0gradients/concat_grad/tuple/control_dependency_1CNN_encoder_cat/catout/Softmax*
T0
v
Cgradients/CNN_encoder_cat/catout/Softmax_grad/Sum/reduction_indicesConst*
valueB :
’’’’’’’’’*
dtype0
Ö
1gradients/CNN_encoder_cat/catout/Softmax_grad/SumSum1gradients/CNN_encoder_cat/catout/Softmax_grad/mulCgradients/CNN_encoder_cat/catout/Softmax_grad/Sum/reduction_indices*

Tidx0*
	keep_dims(*
T0
¦
1gradients/CNN_encoder_cat/catout/Softmax_grad/subSub0gradients/concat_grad/tuple/control_dependency_11gradients/CNN_encoder_cat/catout/Softmax_grad/Sum*
T0

3gradients/CNN_encoder_cat/catout/Softmax_grad/mul_1Mul1gradients/CNN_encoder_cat/catout/Softmax_grad/subCNN_encoder_cat/catout/Softmax*
T0
Ķ
1gradients/CNN_encoder_cat/zout/MatMul_grad/MatMulMatMulDgradients/CNN_encoder_cat/zout/BiasAdd_grad/tuple/control_dependencyCNN_encoder_cat/zout/W/read*
transpose_b(*
T0*
transpose_a( 
Š
3gradients/CNN_encoder_cat/zout/MatMul_grad/MatMul_1MatMulCNN_encoder_cat/zout/ReshapeDgradients/CNN_encoder_cat/zout/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
­
;gradients/CNN_encoder_cat/zout/MatMul_grad/tuple/group_depsNoOp2^gradients/CNN_encoder_cat/zout/MatMul_grad/MatMul4^gradients/CNN_encoder_cat/zout/MatMul_grad/MatMul_1

Cgradients/CNN_encoder_cat/zout/MatMul_grad/tuple/control_dependencyIdentity1gradients/CNN_encoder_cat/zout/MatMul_grad/MatMul<^gradients/CNN_encoder_cat/zout/MatMul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/CNN_encoder_cat/zout/MatMul_grad/MatMul

Egradients/CNN_encoder_cat/zout/MatMul_grad/tuple/control_dependency_1Identity3gradients/CNN_encoder_cat/zout/MatMul_grad/MatMul_1<^gradients/CNN_encoder_cat/zout/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/CNN_encoder_cat/zout/MatMul_grad/MatMul_1

9gradients/CNN_encoder_cat/catout/BiasAdd_grad/BiasAddGradBiasAddGrad3gradients/CNN_encoder_cat/catout/Softmax_grad/mul_1*
T0*
data_formatNHWC
ø
>gradients/CNN_encoder_cat/catout/BiasAdd_grad/tuple/group_depsNoOp:^gradients/CNN_encoder_cat/catout/BiasAdd_grad/BiasAddGrad4^gradients/CNN_encoder_cat/catout/Softmax_grad/mul_1

Fgradients/CNN_encoder_cat/catout/BiasAdd_grad/tuple/control_dependencyIdentity3gradients/CNN_encoder_cat/catout/Softmax_grad/mul_1?^gradients/CNN_encoder_cat/catout/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/CNN_encoder_cat/catout/Softmax_grad/mul_1
§
Hgradients/CNN_encoder_cat/catout/BiasAdd_grad/tuple/control_dependency_1Identity9gradients/CNN_encoder_cat/catout/BiasAdd_grad/BiasAddGrad?^gradients/CNN_encoder_cat/catout/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/CNN_encoder_cat/catout/BiasAdd_grad/BiasAddGrad
x
1gradients/CNN_encoder_cat/zout/Reshape_grad/ShapeShape#CNN_encoder_cat/MaxPool2D_2/MaxPool*
T0*
out_type0
Ķ
3gradients/CNN_encoder_cat/zout/Reshape_grad/ReshapeReshapeCgradients/CNN_encoder_cat/zout/MatMul_grad/tuple/control_dependency1gradients/CNN_encoder_cat/zout/Reshape_grad/Shape*
T0*
Tshape0
Ó
3gradients/CNN_encoder_cat/catout/MatMul_grad/MatMulMatMulFgradients/CNN_encoder_cat/catout/BiasAdd_grad/tuple/control_dependencyCNN_encoder_cat/catout/W/read*
transpose_b(*
T0*
transpose_a( 
Ö
5gradients/CNN_encoder_cat/catout/MatMul_grad/MatMul_1MatMulCNN_encoder_cat/catout/ReshapeFgradients/CNN_encoder_cat/catout/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
³
=gradients/CNN_encoder_cat/catout/MatMul_grad/tuple/group_depsNoOp4^gradients/CNN_encoder_cat/catout/MatMul_grad/MatMul6^gradients/CNN_encoder_cat/catout/MatMul_grad/MatMul_1

Egradients/CNN_encoder_cat/catout/MatMul_grad/tuple/control_dependencyIdentity3gradients/CNN_encoder_cat/catout/MatMul_grad/MatMul>^gradients/CNN_encoder_cat/catout/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/CNN_encoder_cat/catout/MatMul_grad/MatMul

Ggradients/CNN_encoder_cat/catout/MatMul_grad/tuple/control_dependency_1Identity5gradients/CNN_encoder_cat/catout/MatMul_grad/MatMul_1>^gradients/CNN_encoder_cat/catout/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/CNN_encoder_cat/catout/MatMul_grad/MatMul_1
z
3gradients/CNN_encoder_cat/catout/Reshape_grad/ShapeShape#CNN_encoder_cat/MaxPool2D_2/MaxPool*
T0*
out_type0
Ó
5gradients/CNN_encoder_cat/catout/Reshape_grad/ReshapeReshapeEgradients/CNN_encoder_cat/catout/MatMul_grad/tuple/control_dependency3gradients/CNN_encoder_cat/catout/Reshape_grad/Shape*
T0*
Tshape0
Ü
gradients/AddNAddN3gradients/CNN_encoder_cat/zout/Reshape_grad/Reshape5gradients/CNN_encoder_cat/catout/Reshape_grad/Reshape*
T0*F
_class<
:8loc:@gradients/CNN_encoder_cat/zout/Reshape_grad/Reshape*
N
ü
>gradients/CNN_encoder_cat/MaxPool2D_2/MaxPool_grad/MaxPoolGradMaxPoolGradCNN_encoder_cat/Conv2D_2/Tanh#CNN_encoder_cat/MaxPool2D_2/MaxPoolgradients/AddN*
ksize
*
paddingSAME*
T0*
data_formatNHWC*
strides

©
5gradients/CNN_encoder_cat/Conv2D_2/Tanh_grad/TanhGradTanhGradCNN_encoder_cat/Conv2D_2/Tanh>gradients/CNN_encoder_cat/MaxPool2D_2/MaxPool_grad/MaxPoolGrad*
T0
”
;gradients/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients/CNN_encoder_cat/Conv2D_2/Tanh_grad/TanhGrad*
T0*
data_formatNHWC
¾
@gradients/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/group_depsNoOp<^gradients/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/BiasAddGrad6^gradients/CNN_encoder_cat/Conv2D_2/Tanh_grad/TanhGrad
”
Hgradients/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/control_dependencyIdentity5gradients/CNN_encoder_cat/Conv2D_2/Tanh_grad/TanhGradA^gradients/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/CNN_encoder_cat/Conv2D_2/Tanh_grad/TanhGrad
Æ
Jgradients/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/control_dependency_1Identity;gradients/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/BiasAddGradA^gradients/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/BiasAddGrad
§
5gradients/CNN_encoder_cat/Conv2D_2/Conv2D_grad/ShapeNShapeN#CNN_encoder_cat/MaxPool2D_1/MaxPoolCNN_encoder_cat/Conv2D_2/W/read*
T0*
out_type0*
N

Bgradients/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput5gradients/CNN_encoder_cat/Conv2D_2/Conv2D_grad/ShapeNCNN_encoder_cat/Conv2D_2/W/readHgradients/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME

Cgradients/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter#CNN_encoder_cat/MaxPool2D_1/MaxPool7gradients/CNN_encoder_cat/Conv2D_2/Conv2D_grad/ShapeN:1Hgradients/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
Ņ
?gradients/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/group_depsNoOpD^gradients/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropFilterC^gradients/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropInput
¹
Ggradients/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/control_dependencyIdentityBgradients/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropInput@^gradients/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropInput
½
Igradients/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/control_dependency_1IdentityCgradients/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropFilter@^gradients/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropFilter
µ
>gradients/CNN_encoder_cat/MaxPool2D_1/MaxPool_grad/MaxPoolGradMaxPoolGradCNN_encoder_cat/Conv2D_1/Tanh#CNN_encoder_cat/MaxPool2D_1/MaxPoolGgradients/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*
T0*
data_formatNHWC*
strides

©
5gradients/CNN_encoder_cat/Conv2D_1/Tanh_grad/TanhGradTanhGradCNN_encoder_cat/Conv2D_1/Tanh>gradients/CNN_encoder_cat/MaxPool2D_1/MaxPool_grad/MaxPoolGrad*
T0
”
;gradients/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients/CNN_encoder_cat/Conv2D_1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC
¾
@gradients/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/group_depsNoOp<^gradients/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/BiasAddGrad6^gradients/CNN_encoder_cat/Conv2D_1/Tanh_grad/TanhGrad
”
Hgradients/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/control_dependencyIdentity5gradients/CNN_encoder_cat/Conv2D_1/Tanh_grad/TanhGradA^gradients/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients/CNN_encoder_cat/Conv2D_1/Tanh_grad/TanhGrad
Æ
Jgradients/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/control_dependency_1Identity;gradients/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/BiasAddGradA^gradients/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/BiasAddGrad
„
5gradients/CNN_encoder_cat/Conv2D_1/Conv2D_grad/ShapeNShapeN!CNN_encoder_cat/MaxPool2D/MaxPoolCNN_encoder_cat/Conv2D_1/W/read*
T0*
out_type0*
N

Bgradients/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput5gradients/CNN_encoder_cat/Conv2D_1/Conv2D_grad/ShapeNCNN_encoder_cat/Conv2D_1/W/readHgradients/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME

Cgradients/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter!CNN_encoder_cat/MaxPool2D/MaxPool7gradients/CNN_encoder_cat/Conv2D_1/Conv2D_grad/ShapeN:1Hgradients/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
Ņ
?gradients/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/group_depsNoOpD^gradients/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropFilterC^gradients/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropInput
¹
Ggradients/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/control_dependencyIdentityBgradients/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropInput@^gradients/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropInput
½
Igradients/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/control_dependency_1IdentityCgradients/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropFilter@^gradients/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropFilter
Æ
<gradients/CNN_encoder_cat/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradCNN_encoder_cat/Conv2D/Tanh!CNN_encoder_cat/MaxPool2D/MaxPoolGgradients/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*
T0*
data_formatNHWC*
strides

£
3gradients/CNN_encoder_cat/Conv2D/Tanh_grad/TanhGradTanhGradCNN_encoder_cat/Conv2D/Tanh<gradients/CNN_encoder_cat/MaxPool2D/MaxPool_grad/MaxPoolGrad*
T0

9gradients/CNN_encoder_cat/Conv2D/BiasAdd_grad/BiasAddGradBiasAddGrad3gradients/CNN_encoder_cat/Conv2D/Tanh_grad/TanhGrad*
T0*
data_formatNHWC
ø
>gradients/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/group_depsNoOp:^gradients/CNN_encoder_cat/Conv2D/BiasAdd_grad/BiasAddGrad4^gradients/CNN_encoder_cat/Conv2D/Tanh_grad/TanhGrad

Fgradients/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/control_dependencyIdentity3gradients/CNN_encoder_cat/Conv2D/Tanh_grad/TanhGrad?^gradients/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients/CNN_encoder_cat/Conv2D/Tanh_grad/TanhGrad
§
Hgradients/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/control_dependency_1Identity9gradients/CNN_encoder_cat/Conv2D/BiasAdd_grad/BiasAddGrad?^gradients/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/CNN_encoder_cat/Conv2D/BiasAdd_grad/BiasAddGrad

3gradients/CNN_encoder_cat/Conv2D/Conv2D_grad/ShapeNShapeNCNN_encoder_cat/ReshapeCNN_encoder_cat/Conv2D/W/read*
T0*
out_type0*
N

@gradients/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput3gradients/CNN_encoder_cat/Conv2D/Conv2D_grad/ShapeNCNN_encoder_cat/Conv2D/W/readFgradients/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME

Agradients/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterCNN_encoder_cat/Reshape5gradients/CNN_encoder_cat/Conv2D/Conv2D_grad/ShapeN:1Fgradients/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingSAME
Ģ
=gradients/CNN_encoder_cat/Conv2D/Conv2D_grad/tuple/group_depsNoOpB^gradients/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropFilterA^gradients/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropInput
±
Egradients/CNN_encoder_cat/Conv2D/Conv2D_grad/tuple/control_dependencyIdentity@gradients/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropInput>^gradients/CNN_encoder_cat/Conv2D/Conv2D_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropInput
µ
Ggradients/CNN_encoder_cat/Conv2D/Conv2D_grad/tuple/control_dependency_1IdentityAgradients/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropFilter>^gradients/CNN_encoder_cat/Conv2D/Conv2D_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropFilter
o
beta1_power/initial_valueConst*'
_class
loc:@CNN_decoder/Conv2D/W*
valueB
 *fff?*
dtype0

beta1_power
VariableV2*
shape: *
shared_name *'
_class
loc:@CNN_decoder/Conv2D/W*
dtype0*
	container 

beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*'
_class
loc:@CNN_decoder/Conv2D/W*
validate_shape(
[
beta1_power/readIdentitybeta1_power*
T0*'
_class
loc:@CNN_decoder/Conv2D/W
o
beta2_power/initial_valueConst*'
_class
loc:@CNN_decoder/Conv2D/W*
valueB
 *w¾?*
dtype0

beta2_power
VariableV2*
shape: *
shared_name *'
_class
loc:@CNN_decoder/Conv2D/W*
dtype0*
	container 

beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*'
_class
loc:@CNN_decoder/Conv2D/W*
validate_shape(
[
beta2_power/readIdentitybeta2_power*
T0*'
_class
loc:@CNN_decoder/Conv2D/W

/CNN_encoder_cat/Conv2D/W/Adam/Initializer/zerosConst*%
valueB *    *+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
dtype0
¦
CNN_encoder_cat/Conv2D/W/Adam
VariableV2*
shape: *
shared_name *+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
dtype0*
	container 
Ż
$CNN_encoder_cat/Conv2D/W/Adam/AssignAssignCNN_encoder_cat/Conv2D/W/Adam/CNN_encoder_cat/Conv2D/W/Adam/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(

"CNN_encoder_cat/Conv2D/W/Adam/readIdentityCNN_encoder_cat/Conv2D/W/Adam*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W

1CNN_encoder_cat/Conv2D/W/Adam_1/Initializer/zerosConst*%
valueB *    *+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
dtype0
Ø
CNN_encoder_cat/Conv2D/W/Adam_1
VariableV2*
shape: *
shared_name *+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
dtype0*
	container 
ć
&CNN_encoder_cat/Conv2D/W/Adam_1/AssignAssignCNN_encoder_cat/Conv2D/W/Adam_11CNN_encoder_cat/Conv2D/W/Adam_1/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(

$CNN_encoder_cat/Conv2D/W/Adam_1/readIdentityCNN_encoder_cat/Conv2D/W/Adam_1*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W

/CNN_encoder_cat/Conv2D/b/Adam/Initializer/zerosConst*
valueB *    *+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
dtype0

CNN_encoder_cat/Conv2D/b/Adam
VariableV2*
shape: *
shared_name *+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
dtype0*
	container 
Ż
$CNN_encoder_cat/Conv2D/b/Adam/AssignAssignCNN_encoder_cat/Conv2D/b/Adam/CNN_encoder_cat/Conv2D/b/Adam/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
validate_shape(

"CNN_encoder_cat/Conv2D/b/Adam/readIdentityCNN_encoder_cat/Conv2D/b/Adam*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b

1CNN_encoder_cat/Conv2D/b/Adam_1/Initializer/zerosConst*
valueB *    *+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
dtype0

CNN_encoder_cat/Conv2D/b/Adam_1
VariableV2*
shape: *
shared_name *+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
dtype0*
	container 
ć
&CNN_encoder_cat/Conv2D/b/Adam_1/AssignAssignCNN_encoder_cat/Conv2D/b/Adam_11CNN_encoder_cat/Conv2D/b/Adam_1/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
validate_shape(

$CNN_encoder_cat/Conv2D/b/Adam_1/readIdentityCNN_encoder_cat/Conv2D/b/Adam_1*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b
­
ACNN_encoder_cat/Conv2D_1/W/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"              *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0

7CNN_encoder_cat/Conv2D_1/W/Adam/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0
’
1CNN_encoder_cat/Conv2D_1/W/Adam/Initializer/zerosFillACNN_encoder_cat/Conv2D_1/W/Adam/Initializer/zeros/shape_as_tensor7CNN_encoder_cat/Conv2D_1/W/Adam/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W
Ŗ
CNN_encoder_cat/Conv2D_1/W/Adam
VariableV2*
shape:  *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0*
	container 
å
&CNN_encoder_cat/Conv2D_1/W/Adam/AssignAssignCNN_encoder_cat/Conv2D_1/W/Adam1CNN_encoder_cat/Conv2D_1/W/Adam/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
validate_shape(

$CNN_encoder_cat/Conv2D_1/W/Adam/readIdentityCNN_encoder_cat/Conv2D_1/W/Adam*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W
Æ
CCNN_encoder_cat/Conv2D_1/W/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"              *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0

9CNN_encoder_cat/Conv2D_1/W/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0

3CNN_encoder_cat/Conv2D_1/W/Adam_1/Initializer/zerosFillCCNN_encoder_cat/Conv2D_1/W/Adam_1/Initializer/zeros/shape_as_tensor9CNN_encoder_cat/Conv2D_1/W/Adam_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W
¬
!CNN_encoder_cat/Conv2D_1/W/Adam_1
VariableV2*
shape:  *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_1/W/Adam_1/AssignAssign!CNN_encoder_cat/Conv2D_1/W/Adam_13CNN_encoder_cat/Conv2D_1/W/Adam_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
validate_shape(

&CNN_encoder_cat/Conv2D_1/W/Adam_1/readIdentity!CNN_encoder_cat/Conv2D_1/W/Adam_1*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W

1CNN_encoder_cat/Conv2D_1/b/Adam/Initializer/zerosConst*
valueB *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
dtype0

CNN_encoder_cat/Conv2D_1/b/Adam
VariableV2*
shape: *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
dtype0*
	container 
å
&CNN_encoder_cat/Conv2D_1/b/Adam/AssignAssignCNN_encoder_cat/Conv2D_1/b/Adam1CNN_encoder_cat/Conv2D_1/b/Adam/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
validate_shape(

$CNN_encoder_cat/Conv2D_1/b/Adam/readIdentityCNN_encoder_cat/Conv2D_1/b/Adam*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b

3CNN_encoder_cat/Conv2D_1/b/Adam_1/Initializer/zerosConst*
valueB *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
dtype0
 
!CNN_encoder_cat/Conv2D_1/b/Adam_1
VariableV2*
shape: *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_1/b/Adam_1/AssignAssign!CNN_encoder_cat/Conv2D_1/b/Adam_13CNN_encoder_cat/Conv2D_1/b/Adam_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
validate_shape(

&CNN_encoder_cat/Conv2D_1/b/Adam_1/readIdentity!CNN_encoder_cat/Conv2D_1/b/Adam_1*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b
­
ACNN_encoder_cat/Conv2D_2/W/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"              *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0

7CNN_encoder_cat/Conv2D_2/W/Adam/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0
’
1CNN_encoder_cat/Conv2D_2/W/Adam/Initializer/zerosFillACNN_encoder_cat/Conv2D_2/W/Adam/Initializer/zeros/shape_as_tensor7CNN_encoder_cat/Conv2D_2/W/Adam/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W
Ŗ
CNN_encoder_cat/Conv2D_2/W/Adam
VariableV2*
shape:  *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0*
	container 
å
&CNN_encoder_cat/Conv2D_2/W/Adam/AssignAssignCNN_encoder_cat/Conv2D_2/W/Adam1CNN_encoder_cat/Conv2D_2/W/Adam/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
validate_shape(

$CNN_encoder_cat/Conv2D_2/W/Adam/readIdentityCNN_encoder_cat/Conv2D_2/W/Adam*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W
Æ
CCNN_encoder_cat/Conv2D_2/W/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"              *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0

9CNN_encoder_cat/Conv2D_2/W/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0

3CNN_encoder_cat/Conv2D_2/W/Adam_1/Initializer/zerosFillCCNN_encoder_cat/Conv2D_2/W/Adam_1/Initializer/zeros/shape_as_tensor9CNN_encoder_cat/Conv2D_2/W/Adam_1/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W
¬
!CNN_encoder_cat/Conv2D_2/W/Adam_1
VariableV2*
shape:  *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_2/W/Adam_1/AssignAssign!CNN_encoder_cat/Conv2D_2/W/Adam_13CNN_encoder_cat/Conv2D_2/W/Adam_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
validate_shape(

&CNN_encoder_cat/Conv2D_2/W/Adam_1/readIdentity!CNN_encoder_cat/Conv2D_2/W/Adam_1*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W

1CNN_encoder_cat/Conv2D_2/b/Adam/Initializer/zerosConst*
valueB *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
dtype0

CNN_encoder_cat/Conv2D_2/b/Adam
VariableV2*
shape: *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
dtype0*
	container 
å
&CNN_encoder_cat/Conv2D_2/b/Adam/AssignAssignCNN_encoder_cat/Conv2D_2/b/Adam1CNN_encoder_cat/Conv2D_2/b/Adam/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
validate_shape(

$CNN_encoder_cat/Conv2D_2/b/Adam/readIdentityCNN_encoder_cat/Conv2D_2/b/Adam*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b

3CNN_encoder_cat/Conv2D_2/b/Adam_1/Initializer/zerosConst*
valueB *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
dtype0
 
!CNN_encoder_cat/Conv2D_2/b/Adam_1
VariableV2*
shape: *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_2/b/Adam_1/AssignAssign!CNN_encoder_cat/Conv2D_2/b/Adam_13CNN_encoder_cat/Conv2D_2/b/Adam_1/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
validate_shape(

&CNN_encoder_cat/Conv2D_2/b/Adam_1/readIdentity!CNN_encoder_cat/Conv2D_2/b/Adam_1*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b
”
?CNN_encoder_cat/catout/W/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"     *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0

5CNN_encoder_cat/catout/W/Adam/Initializer/zeros/ConstConst*
valueB
 *    *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0
÷
/CNN_encoder_cat/catout/W/Adam/Initializer/zerosFill?CNN_encoder_cat/catout/W/Adam/Initializer/zeros/shape_as_tensor5CNN_encoder_cat/catout/W/Adam/Initializer/zeros/Const*
T0*

index_type0*+
_class!
loc:@CNN_encoder_cat/catout/W

CNN_encoder_cat/catout/W/Adam
VariableV2*
shape:	*
shared_name *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0*
	container 
Ż
$CNN_encoder_cat/catout/W/Adam/AssignAssignCNN_encoder_cat/catout/W/Adam/CNN_encoder_cat/catout/W/Adam/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W*
validate_shape(

"CNN_encoder_cat/catout/W/Adam/readIdentityCNN_encoder_cat/catout/W/Adam*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W
£
ACNN_encoder_cat/catout/W/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"     *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0

7CNN_encoder_cat/catout/W/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0
ż
1CNN_encoder_cat/catout/W/Adam_1/Initializer/zerosFillACNN_encoder_cat/catout/W/Adam_1/Initializer/zeros/shape_as_tensor7CNN_encoder_cat/catout/W/Adam_1/Initializer/zeros/Const*
T0*

index_type0*+
_class!
loc:@CNN_encoder_cat/catout/W
”
CNN_encoder_cat/catout/W/Adam_1
VariableV2*
shape:	*
shared_name *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0*
	container 
ć
&CNN_encoder_cat/catout/W/Adam_1/AssignAssignCNN_encoder_cat/catout/W/Adam_11CNN_encoder_cat/catout/W/Adam_1/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W*
validate_shape(

$CNN_encoder_cat/catout/W/Adam_1/readIdentityCNN_encoder_cat/catout/W/Adam_1*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W

/CNN_encoder_cat/catout/b/Adam/Initializer/zerosConst*
valueB*    *+
_class!
loc:@CNN_encoder_cat/catout/b*
dtype0

CNN_encoder_cat/catout/b/Adam
VariableV2*
shape:*
shared_name *+
_class!
loc:@CNN_encoder_cat/catout/b*
dtype0*
	container 
Ż
$CNN_encoder_cat/catout/b/Adam/AssignAssignCNN_encoder_cat/catout/b/Adam/CNN_encoder_cat/catout/b/Adam/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b*
validate_shape(

"CNN_encoder_cat/catout/b/Adam/readIdentityCNN_encoder_cat/catout/b/Adam*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b

1CNN_encoder_cat/catout/b/Adam_1/Initializer/zerosConst*
valueB*    *+
_class!
loc:@CNN_encoder_cat/catout/b*
dtype0

CNN_encoder_cat/catout/b/Adam_1
VariableV2*
shape:*
shared_name *+
_class!
loc:@CNN_encoder_cat/catout/b*
dtype0*
	container 
ć
&CNN_encoder_cat/catout/b/Adam_1/AssignAssignCNN_encoder_cat/catout/b/Adam_11CNN_encoder_cat/catout/b/Adam_1/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b*
validate_shape(

$CNN_encoder_cat/catout/b/Adam_1/readIdentityCNN_encoder_cat/catout/b/Adam_1*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b

=CNN_encoder_cat/zout/W/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"  2   *)
_class
loc:@CNN_encoder_cat/zout/W*
dtype0

3CNN_encoder_cat/zout/W/Adam/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@CNN_encoder_cat/zout/W*
dtype0
ļ
-CNN_encoder_cat/zout/W/Adam/Initializer/zerosFill=CNN_encoder_cat/zout/W/Adam/Initializer/zeros/shape_as_tensor3CNN_encoder_cat/zout/W/Adam/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@CNN_encoder_cat/zout/W

CNN_encoder_cat/zout/W/Adam
VariableV2*
shape:	2*
shared_name *)
_class
loc:@CNN_encoder_cat/zout/W*
dtype0*
	container 
Õ
"CNN_encoder_cat/zout/W/Adam/AssignAssignCNN_encoder_cat/zout/W/Adam-CNN_encoder_cat/zout/W/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/W*
validate_shape(
}
 CNN_encoder_cat/zout/W/Adam/readIdentityCNN_encoder_cat/zout/W/Adam*
T0*)
_class
loc:@CNN_encoder_cat/zout/W

?CNN_encoder_cat/zout/W/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"  2   *)
_class
loc:@CNN_encoder_cat/zout/W*
dtype0

5CNN_encoder_cat/zout/W/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@CNN_encoder_cat/zout/W*
dtype0
õ
/CNN_encoder_cat/zout/W/Adam_1/Initializer/zerosFill?CNN_encoder_cat/zout/W/Adam_1/Initializer/zeros/shape_as_tensor5CNN_encoder_cat/zout/W/Adam_1/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@CNN_encoder_cat/zout/W

CNN_encoder_cat/zout/W/Adam_1
VariableV2*
shape:	2*
shared_name *)
_class
loc:@CNN_encoder_cat/zout/W*
dtype0*
	container 
Ū
$CNN_encoder_cat/zout/W/Adam_1/AssignAssignCNN_encoder_cat/zout/W/Adam_1/CNN_encoder_cat/zout/W/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/W*
validate_shape(

"CNN_encoder_cat/zout/W/Adam_1/readIdentityCNN_encoder_cat/zout/W/Adam_1*
T0*)
_class
loc:@CNN_encoder_cat/zout/W

-CNN_encoder_cat/zout/b/Adam/Initializer/zerosConst*
valueB2*    *)
_class
loc:@CNN_encoder_cat/zout/b*
dtype0

CNN_encoder_cat/zout/b/Adam
VariableV2*
shape:2*
shared_name *)
_class
loc:@CNN_encoder_cat/zout/b*
dtype0*
	container 
Õ
"CNN_encoder_cat/zout/b/Adam/AssignAssignCNN_encoder_cat/zout/b/Adam-CNN_encoder_cat/zout/b/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/b*
validate_shape(
}
 CNN_encoder_cat/zout/b/Adam/readIdentityCNN_encoder_cat/zout/b/Adam*
T0*)
_class
loc:@CNN_encoder_cat/zout/b

/CNN_encoder_cat/zout/b/Adam_1/Initializer/zerosConst*
valueB2*    *)
_class
loc:@CNN_encoder_cat/zout/b*
dtype0

CNN_encoder_cat/zout/b/Adam_1
VariableV2*
shape:2*
shared_name *)
_class
loc:@CNN_encoder_cat/zout/b*
dtype0*
	container 
Ū
$CNN_encoder_cat/zout/b/Adam_1/AssignAssignCNN_encoder_cat/zout/b/Adam_1/CNN_encoder_cat/zout/b/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/b*
validate_shape(

"CNN_encoder_cat/zout/b/Adam_1/readIdentityCNN_encoder_cat/zout/b/Adam_1*
T0*)
_class
loc:@CNN_encoder_cat/zout/b
©
CCNN_decoder/FullyConnected/W/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"3   Ą  */
_class%
#!loc:@CNN_decoder/FullyConnected/W*
dtype0

9CNN_decoder/FullyConnected/W/Adam/Initializer/zeros/ConstConst*
valueB
 *    */
_class%
#!loc:@CNN_decoder/FullyConnected/W*
dtype0

3CNN_decoder/FullyConnected/W/Adam/Initializer/zerosFillCCNN_decoder/FullyConnected/W/Adam/Initializer/zeros/shape_as_tensor9CNN_decoder/FullyConnected/W/Adam/Initializer/zeros/Const*
T0*

index_type0*/
_class%
#!loc:@CNN_decoder/FullyConnected/W
§
!CNN_decoder/FullyConnected/W/Adam
VariableV2*
shape:	3Ą	*
shared_name */
_class%
#!loc:@CNN_decoder/FullyConnected/W*
dtype0*
	container 
ķ
(CNN_decoder/FullyConnected/W/Adam/AssignAssign!CNN_decoder/FullyConnected/W/Adam3CNN_decoder/FullyConnected/W/Adam/Initializer/zeros*
use_locking(*
T0*/
_class%
#!loc:@CNN_decoder/FullyConnected/W*
validate_shape(

&CNN_decoder/FullyConnected/W/Adam/readIdentity!CNN_decoder/FullyConnected/W/Adam*
T0*/
_class%
#!loc:@CNN_decoder/FullyConnected/W
«
ECNN_decoder/FullyConnected/W/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"3   Ą  */
_class%
#!loc:@CNN_decoder/FullyConnected/W*
dtype0

;CNN_decoder/FullyConnected/W/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    */
_class%
#!loc:@CNN_decoder/FullyConnected/W*
dtype0

5CNN_decoder/FullyConnected/W/Adam_1/Initializer/zerosFillECNN_decoder/FullyConnected/W/Adam_1/Initializer/zeros/shape_as_tensor;CNN_decoder/FullyConnected/W/Adam_1/Initializer/zeros/Const*
T0*

index_type0*/
_class%
#!loc:@CNN_decoder/FullyConnected/W
©
#CNN_decoder/FullyConnected/W/Adam_1
VariableV2*
shape:	3Ą	*
shared_name */
_class%
#!loc:@CNN_decoder/FullyConnected/W*
dtype0*
	container 
ó
*CNN_decoder/FullyConnected/W/Adam_1/AssignAssign#CNN_decoder/FullyConnected/W/Adam_15CNN_decoder/FullyConnected/W/Adam_1/Initializer/zeros*
use_locking(*
T0*/
_class%
#!loc:@CNN_decoder/FullyConnected/W*
validate_shape(

(CNN_decoder/FullyConnected/W/Adam_1/readIdentity#CNN_decoder/FullyConnected/W/Adam_1*
T0*/
_class%
#!loc:@CNN_decoder/FullyConnected/W
£
CCNN_decoder/FullyConnected/b/Adam/Initializer/zeros/shape_as_tensorConst*
valueB:Ą	*/
_class%
#!loc:@CNN_decoder/FullyConnected/b*
dtype0

9CNN_decoder/FullyConnected/b/Adam/Initializer/zeros/ConstConst*
valueB
 *    */
_class%
#!loc:@CNN_decoder/FullyConnected/b*
dtype0

3CNN_decoder/FullyConnected/b/Adam/Initializer/zerosFillCCNN_decoder/FullyConnected/b/Adam/Initializer/zeros/shape_as_tensor9CNN_decoder/FullyConnected/b/Adam/Initializer/zeros/Const*
T0*

index_type0*/
_class%
#!loc:@CNN_decoder/FullyConnected/b
£
!CNN_decoder/FullyConnected/b/Adam
VariableV2*
shape:Ą	*
shared_name */
_class%
#!loc:@CNN_decoder/FullyConnected/b*
dtype0*
	container 
ķ
(CNN_decoder/FullyConnected/b/Adam/AssignAssign!CNN_decoder/FullyConnected/b/Adam3CNN_decoder/FullyConnected/b/Adam/Initializer/zeros*
use_locking(*
T0*/
_class%
#!loc:@CNN_decoder/FullyConnected/b*
validate_shape(

&CNN_decoder/FullyConnected/b/Adam/readIdentity!CNN_decoder/FullyConnected/b/Adam*
T0*/
_class%
#!loc:@CNN_decoder/FullyConnected/b
„
ECNN_decoder/FullyConnected/b/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB:Ą	*/
_class%
#!loc:@CNN_decoder/FullyConnected/b*
dtype0

;CNN_decoder/FullyConnected/b/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    */
_class%
#!loc:@CNN_decoder/FullyConnected/b*
dtype0

5CNN_decoder/FullyConnected/b/Adam_1/Initializer/zerosFillECNN_decoder/FullyConnected/b/Adam_1/Initializer/zeros/shape_as_tensor;CNN_decoder/FullyConnected/b/Adam_1/Initializer/zeros/Const*
T0*

index_type0*/
_class%
#!loc:@CNN_decoder/FullyConnected/b
„
#CNN_decoder/FullyConnected/b/Adam_1
VariableV2*
shape:Ą	*
shared_name */
_class%
#!loc:@CNN_decoder/FullyConnected/b*
dtype0*
	container 
ó
*CNN_decoder/FullyConnected/b/Adam_1/AssignAssign#CNN_decoder/FullyConnected/b/Adam_15CNN_decoder/FullyConnected/b/Adam_1/Initializer/zeros*
use_locking(*
T0*/
_class%
#!loc:@CNN_decoder/FullyConnected/b*
validate_shape(

(CNN_decoder/FullyConnected/b/Adam_1/readIdentity#CNN_decoder/FullyConnected/b/Adam_1*
T0*/
_class%
#!loc:@CNN_decoder/FullyConnected/b
”
;CNN_decoder/Conv2D/W/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"              *'
_class
loc:@CNN_decoder/Conv2D/W*
dtype0

1CNN_decoder/Conv2D/W/Adam/Initializer/zeros/ConstConst*
valueB
 *    *'
_class
loc:@CNN_decoder/Conv2D/W*
dtype0
ē
+CNN_decoder/Conv2D/W/Adam/Initializer/zerosFill;CNN_decoder/Conv2D/W/Adam/Initializer/zeros/shape_as_tensor1CNN_decoder/Conv2D/W/Adam/Initializer/zeros/Const*
T0*

index_type0*'
_class
loc:@CNN_decoder/Conv2D/W

CNN_decoder/Conv2D/W/Adam
VariableV2*
shape:  *
shared_name *'
_class
loc:@CNN_decoder/Conv2D/W*
dtype0*
	container 
Ķ
 CNN_decoder/Conv2D/W/Adam/AssignAssignCNN_decoder/Conv2D/W/Adam+CNN_decoder/Conv2D/W/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@CNN_decoder/Conv2D/W*
validate_shape(
w
CNN_decoder/Conv2D/W/Adam/readIdentityCNN_decoder/Conv2D/W/Adam*
T0*'
_class
loc:@CNN_decoder/Conv2D/W
£
=CNN_decoder/Conv2D/W/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"              *'
_class
loc:@CNN_decoder/Conv2D/W*
dtype0

3CNN_decoder/Conv2D/W/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *'
_class
loc:@CNN_decoder/Conv2D/W*
dtype0
ķ
-CNN_decoder/Conv2D/W/Adam_1/Initializer/zerosFill=CNN_decoder/Conv2D/W/Adam_1/Initializer/zeros/shape_as_tensor3CNN_decoder/Conv2D/W/Adam_1/Initializer/zeros/Const*
T0*

index_type0*'
_class
loc:@CNN_decoder/Conv2D/W
 
CNN_decoder/Conv2D/W/Adam_1
VariableV2*
shape:  *
shared_name *'
_class
loc:@CNN_decoder/Conv2D/W*
dtype0*
	container 
Ó
"CNN_decoder/Conv2D/W/Adam_1/AssignAssignCNN_decoder/Conv2D/W/Adam_1-CNN_decoder/Conv2D/W/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@CNN_decoder/Conv2D/W*
validate_shape(
{
 CNN_decoder/Conv2D/W/Adam_1/readIdentityCNN_decoder/Conv2D/W/Adam_1*
T0*'
_class
loc:@CNN_decoder/Conv2D/W

+CNN_decoder/Conv2D/b/Adam/Initializer/zerosConst*
valueB *    *'
_class
loc:@CNN_decoder/Conv2D/b*
dtype0

CNN_decoder/Conv2D/b/Adam
VariableV2*
shape: *
shared_name *'
_class
loc:@CNN_decoder/Conv2D/b*
dtype0*
	container 
Ķ
 CNN_decoder/Conv2D/b/Adam/AssignAssignCNN_decoder/Conv2D/b/Adam+CNN_decoder/Conv2D/b/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@CNN_decoder/Conv2D/b*
validate_shape(
w
CNN_decoder/Conv2D/b/Adam/readIdentityCNN_decoder/Conv2D/b/Adam*
T0*'
_class
loc:@CNN_decoder/Conv2D/b

-CNN_decoder/Conv2D/b/Adam_1/Initializer/zerosConst*
valueB *    *'
_class
loc:@CNN_decoder/Conv2D/b*
dtype0

CNN_decoder/Conv2D/b/Adam_1
VariableV2*
shape: *
shared_name *'
_class
loc:@CNN_decoder/Conv2D/b*
dtype0*
	container 
Ó
"CNN_decoder/Conv2D/b/Adam_1/AssignAssignCNN_decoder/Conv2D/b/Adam_1-CNN_decoder/Conv2D/b/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@CNN_decoder/Conv2D/b*
validate_shape(
{
 CNN_decoder/Conv2D/b/Adam_1/readIdentityCNN_decoder/Conv2D/b/Adam_1*
T0*'
_class
loc:@CNN_decoder/Conv2D/b
„
=CNN_decoder/Conv2D_1/W/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"          @   *)
_class
loc:@CNN_decoder/Conv2D_1/W*
dtype0

3CNN_decoder/Conv2D_1/W/Adam/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@CNN_decoder/Conv2D_1/W*
dtype0
ļ
-CNN_decoder/Conv2D_1/W/Adam/Initializer/zerosFill=CNN_decoder/Conv2D_1/W/Adam/Initializer/zeros/shape_as_tensor3CNN_decoder/Conv2D_1/W/Adam/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@CNN_decoder/Conv2D_1/W
¢
CNN_decoder/Conv2D_1/W/Adam
VariableV2*
shape: @*
shared_name *)
_class
loc:@CNN_decoder/Conv2D_1/W*
dtype0*
	container 
Õ
"CNN_decoder/Conv2D_1/W/Adam/AssignAssignCNN_decoder/Conv2D_1/W/Adam-CNN_decoder/Conv2D_1/W/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@CNN_decoder/Conv2D_1/W*
validate_shape(
}
 CNN_decoder/Conv2D_1/W/Adam/readIdentityCNN_decoder/Conv2D_1/W/Adam*
T0*)
_class
loc:@CNN_decoder/Conv2D_1/W
§
?CNN_decoder/Conv2D_1/W/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"          @   *)
_class
loc:@CNN_decoder/Conv2D_1/W*
dtype0

5CNN_decoder/Conv2D_1/W/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@CNN_decoder/Conv2D_1/W*
dtype0
õ
/CNN_decoder/Conv2D_1/W/Adam_1/Initializer/zerosFill?CNN_decoder/Conv2D_1/W/Adam_1/Initializer/zeros/shape_as_tensor5CNN_decoder/Conv2D_1/W/Adam_1/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@CNN_decoder/Conv2D_1/W
¤
CNN_decoder/Conv2D_1/W/Adam_1
VariableV2*
shape: @*
shared_name *)
_class
loc:@CNN_decoder/Conv2D_1/W*
dtype0*
	container 
Ū
$CNN_decoder/Conv2D_1/W/Adam_1/AssignAssignCNN_decoder/Conv2D_1/W/Adam_1/CNN_decoder/Conv2D_1/W/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@CNN_decoder/Conv2D_1/W*
validate_shape(

"CNN_decoder/Conv2D_1/W/Adam_1/readIdentityCNN_decoder/Conv2D_1/W/Adam_1*
T0*)
_class
loc:@CNN_decoder/Conv2D_1/W

-CNN_decoder/Conv2D_1/b/Adam/Initializer/zerosConst*
valueB@*    *)
_class
loc:@CNN_decoder/Conv2D_1/b*
dtype0

CNN_decoder/Conv2D_1/b/Adam
VariableV2*
shape:@*
shared_name *)
_class
loc:@CNN_decoder/Conv2D_1/b*
dtype0*
	container 
Õ
"CNN_decoder/Conv2D_1/b/Adam/AssignAssignCNN_decoder/Conv2D_1/b/Adam-CNN_decoder/Conv2D_1/b/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@CNN_decoder/Conv2D_1/b*
validate_shape(
}
 CNN_decoder/Conv2D_1/b/Adam/readIdentityCNN_decoder/Conv2D_1/b/Adam*
T0*)
_class
loc:@CNN_decoder/Conv2D_1/b

/CNN_decoder/Conv2D_1/b/Adam_1/Initializer/zerosConst*
valueB@*    *)
_class
loc:@CNN_decoder/Conv2D_1/b*
dtype0

CNN_decoder/Conv2D_1/b/Adam_1
VariableV2*
shape:@*
shared_name *)
_class
loc:@CNN_decoder/Conv2D_1/b*
dtype0*
	container 
Ū
$CNN_decoder/Conv2D_1/b/Adam_1/AssignAssignCNN_decoder/Conv2D_1/b/Adam_1/CNN_decoder/Conv2D_1/b/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@CNN_decoder/Conv2D_1/b*
validate_shape(

"CNN_decoder/Conv2D_1/b/Adam_1/readIdentityCNN_decoder/Conv2D_1/b/Adam_1*
T0*)
_class
loc:@CNN_decoder/Conv2D_1/b
„
=CNN_decoder/Conv2D_2/W/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"      @   @   *)
_class
loc:@CNN_decoder/Conv2D_2/W*
dtype0

3CNN_decoder/Conv2D_2/W/Adam/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@CNN_decoder/Conv2D_2/W*
dtype0
ļ
-CNN_decoder/Conv2D_2/W/Adam/Initializer/zerosFill=CNN_decoder/Conv2D_2/W/Adam/Initializer/zeros/shape_as_tensor3CNN_decoder/Conv2D_2/W/Adam/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@CNN_decoder/Conv2D_2/W
¢
CNN_decoder/Conv2D_2/W/Adam
VariableV2*
shape:@@*
shared_name *)
_class
loc:@CNN_decoder/Conv2D_2/W*
dtype0*
	container 
Õ
"CNN_decoder/Conv2D_2/W/Adam/AssignAssignCNN_decoder/Conv2D_2/W/Adam-CNN_decoder/Conv2D_2/W/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@CNN_decoder/Conv2D_2/W*
validate_shape(
}
 CNN_decoder/Conv2D_2/W/Adam/readIdentityCNN_decoder/Conv2D_2/W/Adam*
T0*)
_class
loc:@CNN_decoder/Conv2D_2/W
§
?CNN_decoder/Conv2D_2/W/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"      @   @   *)
_class
loc:@CNN_decoder/Conv2D_2/W*
dtype0

5CNN_decoder/Conv2D_2/W/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@CNN_decoder/Conv2D_2/W*
dtype0
õ
/CNN_decoder/Conv2D_2/W/Adam_1/Initializer/zerosFill?CNN_decoder/Conv2D_2/W/Adam_1/Initializer/zeros/shape_as_tensor5CNN_decoder/Conv2D_2/W/Adam_1/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@CNN_decoder/Conv2D_2/W
¤
CNN_decoder/Conv2D_2/W/Adam_1
VariableV2*
shape:@@*
shared_name *)
_class
loc:@CNN_decoder/Conv2D_2/W*
dtype0*
	container 
Ū
$CNN_decoder/Conv2D_2/W/Adam_1/AssignAssignCNN_decoder/Conv2D_2/W/Adam_1/CNN_decoder/Conv2D_2/W/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@CNN_decoder/Conv2D_2/W*
validate_shape(

"CNN_decoder/Conv2D_2/W/Adam_1/readIdentityCNN_decoder/Conv2D_2/W/Adam_1*
T0*)
_class
loc:@CNN_decoder/Conv2D_2/W

-CNN_decoder/Conv2D_2/b/Adam/Initializer/zerosConst*
valueB@*    *)
_class
loc:@CNN_decoder/Conv2D_2/b*
dtype0

CNN_decoder/Conv2D_2/b/Adam
VariableV2*
shape:@*
shared_name *)
_class
loc:@CNN_decoder/Conv2D_2/b*
dtype0*
	container 
Õ
"CNN_decoder/Conv2D_2/b/Adam/AssignAssignCNN_decoder/Conv2D_2/b/Adam-CNN_decoder/Conv2D_2/b/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@CNN_decoder/Conv2D_2/b*
validate_shape(
}
 CNN_decoder/Conv2D_2/b/Adam/readIdentityCNN_decoder/Conv2D_2/b/Adam*
T0*)
_class
loc:@CNN_decoder/Conv2D_2/b

/CNN_decoder/Conv2D_2/b/Adam_1/Initializer/zerosConst*
valueB@*    *)
_class
loc:@CNN_decoder/Conv2D_2/b*
dtype0

CNN_decoder/Conv2D_2/b/Adam_1
VariableV2*
shape:@*
shared_name *)
_class
loc:@CNN_decoder/Conv2D_2/b*
dtype0*
	container 
Ū
$CNN_decoder/Conv2D_2/b/Adam_1/AssignAssignCNN_decoder/Conv2D_2/b/Adam_1/CNN_decoder/Conv2D_2/b/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@CNN_decoder/Conv2D_2/b*
validate_shape(

"CNN_decoder/Conv2D_2/b/Adam_1/readIdentityCNN_decoder/Conv2D_2/b/Adam_1*
T0*)
_class
loc:@CNN_decoder/Conv2D_2/b

+CNN_decoder/sigout/W/Adam/Initializer/zerosConst*%
valueB@*    *'
_class
loc:@CNN_decoder/sigout/W*
dtype0

CNN_decoder/sigout/W/Adam
VariableV2*
shape:@*
shared_name *'
_class
loc:@CNN_decoder/sigout/W*
dtype0*
	container 
Ķ
 CNN_decoder/sigout/W/Adam/AssignAssignCNN_decoder/sigout/W/Adam+CNN_decoder/sigout/W/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@CNN_decoder/sigout/W*
validate_shape(
w
CNN_decoder/sigout/W/Adam/readIdentityCNN_decoder/sigout/W/Adam*
T0*'
_class
loc:@CNN_decoder/sigout/W

-CNN_decoder/sigout/W/Adam_1/Initializer/zerosConst*%
valueB@*    *'
_class
loc:@CNN_decoder/sigout/W*
dtype0
 
CNN_decoder/sigout/W/Adam_1
VariableV2*
shape:@*
shared_name *'
_class
loc:@CNN_decoder/sigout/W*
dtype0*
	container 
Ó
"CNN_decoder/sigout/W/Adam_1/AssignAssignCNN_decoder/sigout/W/Adam_1-CNN_decoder/sigout/W/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@CNN_decoder/sigout/W*
validate_shape(
{
 CNN_decoder/sigout/W/Adam_1/readIdentityCNN_decoder/sigout/W/Adam_1*
T0*'
_class
loc:@CNN_decoder/sigout/W

+CNN_decoder/sigout/b/Adam/Initializer/zerosConst*
valueB*    *'
_class
loc:@CNN_decoder/sigout/b*
dtype0

CNN_decoder/sigout/b/Adam
VariableV2*
shape:*
shared_name *'
_class
loc:@CNN_decoder/sigout/b*
dtype0*
	container 
Ķ
 CNN_decoder/sigout/b/Adam/AssignAssignCNN_decoder/sigout/b/Adam+CNN_decoder/sigout/b/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@CNN_decoder/sigout/b*
validate_shape(
w
CNN_decoder/sigout/b/Adam/readIdentityCNN_decoder/sigout/b/Adam*
T0*'
_class
loc:@CNN_decoder/sigout/b

-CNN_decoder/sigout/b/Adam_1/Initializer/zerosConst*
valueB*    *'
_class
loc:@CNN_decoder/sigout/b*
dtype0

CNN_decoder/sigout/b/Adam_1
VariableV2*
shape:*
shared_name *'
_class
loc:@CNN_decoder/sigout/b*
dtype0*
	container 
Ó
"CNN_decoder/sigout/b/Adam_1/AssignAssignCNN_decoder/sigout/b/Adam_1-CNN_decoder/sigout/b/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@CNN_decoder/sigout/b*
validate_shape(
{
 CNN_decoder/sigout/b/Adam_1/readIdentityCNN_decoder/sigout/b/Adam_1*
T0*'
_class
loc:@CNN_decoder/sigout/b

=CNN_decoder/sigout_1/W/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@CNN_decoder/sigout_1/W*
dtype0

3CNN_decoder/sigout_1/W/Adam/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@CNN_decoder/sigout_1/W*
dtype0
ļ
-CNN_decoder/sigout_1/W/Adam/Initializer/zerosFill=CNN_decoder/sigout_1/W/Adam/Initializer/zeros/shape_as_tensor3CNN_decoder/sigout_1/W/Adam/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@CNN_decoder/sigout_1/W

CNN_decoder/sigout_1/W/Adam
VariableV2*
shape:
  *
shared_name *)
_class
loc:@CNN_decoder/sigout_1/W*
dtype0*
	container 
Õ
"CNN_decoder/sigout_1/W/Adam/AssignAssignCNN_decoder/sigout_1/W/Adam-CNN_decoder/sigout_1/W/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@CNN_decoder/sigout_1/W*
validate_shape(
}
 CNN_decoder/sigout_1/W/Adam/readIdentityCNN_decoder/sigout_1/W/Adam*
T0*)
_class
loc:@CNN_decoder/sigout_1/W

?CNN_decoder/sigout_1/W/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@CNN_decoder/sigout_1/W*
dtype0

5CNN_decoder/sigout_1/W/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@CNN_decoder/sigout_1/W*
dtype0
õ
/CNN_decoder/sigout_1/W/Adam_1/Initializer/zerosFill?CNN_decoder/sigout_1/W/Adam_1/Initializer/zeros/shape_as_tensor5CNN_decoder/sigout_1/W/Adam_1/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@CNN_decoder/sigout_1/W

CNN_decoder/sigout_1/W/Adam_1
VariableV2*
shape:
  *
shared_name *)
_class
loc:@CNN_decoder/sigout_1/W*
dtype0*
	container 
Ū
$CNN_decoder/sigout_1/W/Adam_1/AssignAssignCNN_decoder/sigout_1/W/Adam_1/CNN_decoder/sigout_1/W/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@CNN_decoder/sigout_1/W*
validate_shape(

"CNN_decoder/sigout_1/W/Adam_1/readIdentityCNN_decoder/sigout_1/W/Adam_1*
T0*)
_class
loc:@CNN_decoder/sigout_1/W

=CNN_decoder/sigout_1/b/Adam/Initializer/zeros/shape_as_tensorConst*
valueB: *)
_class
loc:@CNN_decoder/sigout_1/b*
dtype0

3CNN_decoder/sigout_1/b/Adam/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@CNN_decoder/sigout_1/b*
dtype0
ļ
-CNN_decoder/sigout_1/b/Adam/Initializer/zerosFill=CNN_decoder/sigout_1/b/Adam/Initializer/zeros/shape_as_tensor3CNN_decoder/sigout_1/b/Adam/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@CNN_decoder/sigout_1/b

CNN_decoder/sigout_1/b/Adam
VariableV2*
shape: *
shared_name *)
_class
loc:@CNN_decoder/sigout_1/b*
dtype0*
	container 
Õ
"CNN_decoder/sigout_1/b/Adam/AssignAssignCNN_decoder/sigout_1/b/Adam-CNN_decoder/sigout_1/b/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@CNN_decoder/sigout_1/b*
validate_shape(
}
 CNN_decoder/sigout_1/b/Adam/readIdentityCNN_decoder/sigout_1/b/Adam*
T0*)
_class
loc:@CNN_decoder/sigout_1/b

?CNN_decoder/sigout_1/b/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB: *)
_class
loc:@CNN_decoder/sigout_1/b*
dtype0

5CNN_decoder/sigout_1/b/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@CNN_decoder/sigout_1/b*
dtype0
õ
/CNN_decoder/sigout_1/b/Adam_1/Initializer/zerosFill?CNN_decoder/sigout_1/b/Adam_1/Initializer/zeros/shape_as_tensor5CNN_decoder/sigout_1/b/Adam_1/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@CNN_decoder/sigout_1/b

CNN_decoder/sigout_1/b/Adam_1
VariableV2*
shape: *
shared_name *)
_class
loc:@CNN_decoder/sigout_1/b*
dtype0*
	container 
Ū
$CNN_decoder/sigout_1/b/Adam_1/AssignAssignCNN_decoder/sigout_1/b/Adam_1/CNN_decoder/sigout_1/b/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@CNN_decoder/sigout_1/b*
validate_shape(

"CNN_decoder/sigout_1/b/Adam_1/readIdentityCNN_decoder/sigout_1/b/Adam_1*
T0*)
_class
loc:@CNN_decoder/sigout_1/b
?
Adam/learning_rateConst*
valueB
 *·Q8*
dtype0
7

Adam/beta1Const*
valueB
 *fff?*
dtype0
7

Adam/beta2Const*
valueB
 *w¾?*
dtype0
9
Adam/epsilonConst*
valueB
 *wĢ+2*
dtype0

.Adam/update_CNN_encoder_cat/Conv2D/W/ApplyAdam	ApplyAdamCNN_encoder_cat/Conv2D/WCNN_encoder_cat/Conv2D/W/AdamCNN_encoder_cat/Conv2D/W/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonGgradients/CNN_encoder_cat/Conv2D/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
use_nesterov( 

.Adam/update_CNN_encoder_cat/Conv2D/b/ApplyAdam	ApplyAdamCNN_encoder_cat/Conv2D/bCNN_encoder_cat/Conv2D/b/AdamCNN_encoder_cat/Conv2D/b/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonHgradients/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
use_nesterov( 
„
0Adam/update_CNN_encoder_cat/Conv2D_1/W/ApplyAdam	ApplyAdamCNN_encoder_cat/Conv2D_1/WCNN_encoder_cat/Conv2D_1/W/Adam!CNN_encoder_cat/Conv2D_1/W/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonIgradients/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
use_nesterov( 
¦
0Adam/update_CNN_encoder_cat/Conv2D_1/b/ApplyAdam	ApplyAdamCNN_encoder_cat/Conv2D_1/bCNN_encoder_cat/Conv2D_1/b/Adam!CNN_encoder_cat/Conv2D_1/b/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonJgradients/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
use_nesterov( 
„
0Adam/update_CNN_encoder_cat/Conv2D_2/W/ApplyAdam	ApplyAdamCNN_encoder_cat/Conv2D_2/WCNN_encoder_cat/Conv2D_2/W/Adam!CNN_encoder_cat/Conv2D_2/W/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonIgradients/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
use_nesterov( 
¦
0Adam/update_CNN_encoder_cat/Conv2D_2/b/ApplyAdam	ApplyAdamCNN_encoder_cat/Conv2D_2/bCNN_encoder_cat/Conv2D_2/b/Adam!CNN_encoder_cat/Conv2D_2/b/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonJgradients/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
use_nesterov( 

.Adam/update_CNN_encoder_cat/catout/W/ApplyAdam	ApplyAdamCNN_encoder_cat/catout/WCNN_encoder_cat/catout/W/AdamCNN_encoder_cat/catout/W/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonGgradients/CNN_encoder_cat/catout/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*+
_class!
loc:@CNN_encoder_cat/catout/W*
use_nesterov( 

.Adam/update_CNN_encoder_cat/catout/b/ApplyAdam	ApplyAdamCNN_encoder_cat/catout/bCNN_encoder_cat/catout/b/AdamCNN_encoder_cat/catout/b/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonHgradients/CNN_encoder_cat/catout/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*+
_class!
loc:@CNN_encoder_cat/catout/b*
use_nesterov( 

,Adam/update_CNN_encoder_cat/zout/W/ApplyAdam	ApplyAdamCNN_encoder_cat/zout/WCNN_encoder_cat/zout/W/AdamCNN_encoder_cat/zout/W/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonEgradients/CNN_encoder_cat/zout/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@CNN_encoder_cat/zout/W*
use_nesterov( 

,Adam/update_CNN_encoder_cat/zout/b/ApplyAdam	ApplyAdamCNN_encoder_cat/zout/bCNN_encoder_cat/zout/b/AdamCNN_encoder_cat/zout/b/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonFgradients/CNN_encoder_cat/zout/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@CNN_encoder_cat/zout/b*
use_nesterov( 
±
2Adam/update_CNN_decoder/FullyConnected/W/ApplyAdam	ApplyAdamCNN_decoder/FullyConnected/W!CNN_decoder/FullyConnected/W/Adam#CNN_decoder/FullyConnected/W/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonKgradients/CNN_decoder/FullyConnected/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*/
_class%
#!loc:@CNN_decoder/FullyConnected/W*
use_nesterov( 
²
2Adam/update_CNN_decoder/FullyConnected/b/ApplyAdam	ApplyAdamCNN_decoder/FullyConnected/b!CNN_decoder/FullyConnected/b/Adam#CNN_decoder/FullyConnected/b/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonLgradients/CNN_decoder/FullyConnected/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*/
_class%
#!loc:@CNN_decoder/FullyConnected/b*
use_nesterov( 

*Adam/update_CNN_decoder/Conv2D/W/ApplyAdam	ApplyAdamCNN_decoder/Conv2D/WCNN_decoder/Conv2D/W/AdamCNN_decoder/Conv2D/W/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonCgradients/CNN_decoder/Conv2D/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@CNN_decoder/Conv2D/W*
use_nesterov( 

*Adam/update_CNN_decoder/Conv2D/b/ApplyAdam	ApplyAdamCNN_decoder/Conv2D/bCNN_decoder/Conv2D/b/AdamCNN_decoder/Conv2D/b/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonDgradients/CNN_decoder/Conv2D/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@CNN_decoder/Conv2D/b*
use_nesterov( 

,Adam/update_CNN_decoder/Conv2D_1/W/ApplyAdam	ApplyAdamCNN_decoder/Conv2D_1/WCNN_decoder/Conv2D_1/W/AdamCNN_decoder/Conv2D_1/W/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonEgradients/CNN_decoder/Conv2D_1/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@CNN_decoder/Conv2D_1/W*
use_nesterov( 

,Adam/update_CNN_decoder/Conv2D_1/b/ApplyAdam	ApplyAdamCNN_decoder/Conv2D_1/bCNN_decoder/Conv2D_1/b/AdamCNN_decoder/Conv2D_1/b/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonFgradients/CNN_decoder/Conv2D_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@CNN_decoder/Conv2D_1/b*
use_nesterov( 

,Adam/update_CNN_decoder/Conv2D_2/W/ApplyAdam	ApplyAdamCNN_decoder/Conv2D_2/WCNN_decoder/Conv2D_2/W/AdamCNN_decoder/Conv2D_2/W/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonEgradients/CNN_decoder/Conv2D_2/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@CNN_decoder/Conv2D_2/W*
use_nesterov( 

,Adam/update_CNN_decoder/Conv2D_2/b/ApplyAdam	ApplyAdamCNN_decoder/Conv2D_2/bCNN_decoder/Conv2D_2/b/AdamCNN_decoder/Conv2D_2/b/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonFgradients/CNN_decoder/Conv2D_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@CNN_decoder/Conv2D_2/b*
use_nesterov( 

*Adam/update_CNN_decoder/sigout/W/ApplyAdam	ApplyAdamCNN_decoder/sigout/WCNN_decoder/sigout/W/AdamCNN_decoder/sigout/W/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonCgradients/CNN_decoder/sigout/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@CNN_decoder/sigout/W*
use_nesterov( 

*Adam/update_CNN_decoder/sigout/b/ApplyAdam	ApplyAdamCNN_decoder/sigout/bCNN_decoder/sigout/b/AdamCNN_decoder/sigout/b/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonDgradients/CNN_decoder/sigout/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@CNN_decoder/sigout/b*
use_nesterov( 

,Adam/update_CNN_decoder/sigout_1/W/ApplyAdam	ApplyAdamCNN_decoder/sigout_1/WCNN_decoder/sigout_1/W/AdamCNN_decoder/sigout_1/W/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonEgradients/CNN_decoder/sigout_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@CNN_decoder/sigout_1/W*
use_nesterov( 

,Adam/update_CNN_decoder/sigout_1/b/ApplyAdam	ApplyAdamCNN_decoder/sigout_1/bCNN_decoder/sigout_1/b/AdamCNN_decoder/sigout_1/b/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonFgradients/CNN_decoder/sigout_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@CNN_decoder/sigout_1/b*
use_nesterov( 
	
Adam/mulMulbeta1_power/read
Adam/beta1+^Adam/update_CNN_decoder/Conv2D/W/ApplyAdam+^Adam/update_CNN_decoder/Conv2D/b/ApplyAdam-^Adam/update_CNN_decoder/Conv2D_1/W/ApplyAdam-^Adam/update_CNN_decoder/Conv2D_1/b/ApplyAdam-^Adam/update_CNN_decoder/Conv2D_2/W/ApplyAdam-^Adam/update_CNN_decoder/Conv2D_2/b/ApplyAdam3^Adam/update_CNN_decoder/FullyConnected/W/ApplyAdam3^Adam/update_CNN_decoder/FullyConnected/b/ApplyAdam+^Adam/update_CNN_decoder/sigout/W/ApplyAdam+^Adam/update_CNN_decoder/sigout/b/ApplyAdam-^Adam/update_CNN_decoder/sigout_1/W/ApplyAdam-^Adam/update_CNN_decoder/sigout_1/b/ApplyAdam/^Adam/update_CNN_encoder_cat/Conv2D/W/ApplyAdam/^Adam/update_CNN_encoder_cat/Conv2D/b/ApplyAdam1^Adam/update_CNN_encoder_cat/Conv2D_1/W/ApplyAdam1^Adam/update_CNN_encoder_cat/Conv2D_1/b/ApplyAdam1^Adam/update_CNN_encoder_cat/Conv2D_2/W/ApplyAdam1^Adam/update_CNN_encoder_cat/Conv2D_2/b/ApplyAdam/^Adam/update_CNN_encoder_cat/catout/W/ApplyAdam/^Adam/update_CNN_encoder_cat/catout/b/ApplyAdam-^Adam/update_CNN_encoder_cat/zout/W/ApplyAdam-^Adam/update_CNN_encoder_cat/zout/b/ApplyAdam*
T0*'
_class
loc:@CNN_decoder/Conv2D/W

Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*'
_class
loc:@CNN_decoder/Conv2D/W*
validate_shape(
	

Adam/mul_1Mulbeta2_power/read
Adam/beta2+^Adam/update_CNN_decoder/Conv2D/W/ApplyAdam+^Adam/update_CNN_decoder/Conv2D/b/ApplyAdam-^Adam/update_CNN_decoder/Conv2D_1/W/ApplyAdam-^Adam/update_CNN_decoder/Conv2D_1/b/ApplyAdam-^Adam/update_CNN_decoder/Conv2D_2/W/ApplyAdam-^Adam/update_CNN_decoder/Conv2D_2/b/ApplyAdam3^Adam/update_CNN_decoder/FullyConnected/W/ApplyAdam3^Adam/update_CNN_decoder/FullyConnected/b/ApplyAdam+^Adam/update_CNN_decoder/sigout/W/ApplyAdam+^Adam/update_CNN_decoder/sigout/b/ApplyAdam-^Adam/update_CNN_decoder/sigout_1/W/ApplyAdam-^Adam/update_CNN_decoder/sigout_1/b/ApplyAdam/^Adam/update_CNN_encoder_cat/Conv2D/W/ApplyAdam/^Adam/update_CNN_encoder_cat/Conv2D/b/ApplyAdam1^Adam/update_CNN_encoder_cat/Conv2D_1/W/ApplyAdam1^Adam/update_CNN_encoder_cat/Conv2D_1/b/ApplyAdam1^Adam/update_CNN_encoder_cat/Conv2D_2/W/ApplyAdam1^Adam/update_CNN_encoder_cat/Conv2D_2/b/ApplyAdam/^Adam/update_CNN_encoder_cat/catout/W/ApplyAdam/^Adam/update_CNN_encoder_cat/catout/b/ApplyAdam-^Adam/update_CNN_encoder_cat/zout/W/ApplyAdam-^Adam/update_CNN_encoder_cat/zout/b/ApplyAdam*
T0*'
_class
loc:@CNN_decoder/Conv2D/W

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*'
_class
loc:@CNN_decoder/Conv2D/W*
validate_shape(
Š
AdamNoOp^Adam/Assign^Adam/Assign_1+^Adam/update_CNN_decoder/Conv2D/W/ApplyAdam+^Adam/update_CNN_decoder/Conv2D/b/ApplyAdam-^Adam/update_CNN_decoder/Conv2D_1/W/ApplyAdam-^Adam/update_CNN_decoder/Conv2D_1/b/ApplyAdam-^Adam/update_CNN_decoder/Conv2D_2/W/ApplyAdam-^Adam/update_CNN_decoder/Conv2D_2/b/ApplyAdam3^Adam/update_CNN_decoder/FullyConnected/W/ApplyAdam3^Adam/update_CNN_decoder/FullyConnected/b/ApplyAdam+^Adam/update_CNN_decoder/sigout/W/ApplyAdam+^Adam/update_CNN_decoder/sigout/b/ApplyAdam-^Adam/update_CNN_decoder/sigout_1/W/ApplyAdam-^Adam/update_CNN_decoder/sigout_1/b/ApplyAdam/^Adam/update_CNN_encoder_cat/Conv2D/W/ApplyAdam/^Adam/update_CNN_encoder_cat/Conv2D/b/ApplyAdam1^Adam/update_CNN_encoder_cat/Conv2D_1/W/ApplyAdam1^Adam/update_CNN_encoder_cat/Conv2D_1/b/ApplyAdam1^Adam/update_CNN_encoder_cat/Conv2D_2/W/ApplyAdam1^Adam/update_CNN_encoder_cat/Conv2D_2/b/ApplyAdam/^Adam/update_CNN_encoder_cat/catout/W/ApplyAdam/^Adam/update_CNN_encoder_cat/catout/b/ApplyAdam-^Adam/update_CNN_encoder_cat/zout/W/ApplyAdam-^Adam/update_CNN_encoder_cat/zout/b/ApplyAdam
:
gradients_1/ShapeConst*
valueB *
dtype0
B
gradients_1/grad_ys_0Const*
valueB
 *  ?*
dtype0
]
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*

index_type0
N
%gradients_1/Mean_7_grad/Reshape/shapeConst*
valueB *
dtype0
z
gradients_1/Mean_7_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_7_grad/Reshape/shape*
T0*
Tshape0
F
gradients_1/Mean_7_grad/ConstConst*
valueB *
dtype0

gradients_1/Mean_7_grad/TileTilegradients_1/Mean_7_grad/Reshapegradients_1/Mean_7_grad/Const*

Tmultiples0*
T0
L
gradients_1/Mean_7_grad/Const_1Const*
valueB
 *  ?*
dtype0
r
gradients_1/Mean_7_grad/truedivRealDivgradients_1/Mean_7_grad/Tilegradients_1/Mean_7_grad/Const_1*
T0
Q
'gradients_1/add_2_grad/tuple/group_depsNoOp ^gradients_1/Mean_7_grad/truediv
Ć
/gradients_1/add_2_grad/tuple/control_dependencyIdentitygradients_1/Mean_7_grad/truediv(^gradients_1/add_2_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_1/Mean_7_grad/truediv
Å
1gradients_1/add_2_grad/tuple/control_dependency_1Identitygradients_1/Mean_7_grad/truediv(^gradients_1/add_2_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_1/Mean_7_grad/truediv
_
%gradients_1/add_grad/tuple/group_depsNoOp0^gradients_1/add_2_grad/tuple/control_dependency
Ļ
-gradients_1/add_grad/tuple/control_dependencyIdentity/gradients_1/add_2_grad/tuple/control_dependency&^gradients_1/add_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_1/Mean_7_grad/truediv
Ń
/gradients_1/add_grad/tuple/control_dependency_1Identity/gradients_1/add_2_grad/tuple/control_dependency&^gradients_1/add_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_1/Mean_7_grad/truediv
c
'gradients_1/add_1_grad/tuple/group_depsNoOp2^gradients_1/add_2_grad/tuple/control_dependency_1
Õ
/gradients_1/add_1_grad/tuple/control_dependencyIdentity1gradients_1/add_2_grad/tuple/control_dependency_1(^gradients_1/add_1_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_1/Mean_7_grad/truediv
×
1gradients_1/add_1_grad/tuple/control_dependency_1Identity1gradients_1/add_2_grad/tuple/control_dependency_1(^gradients_1/add_1_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_1/Mean_7_grad/truediv
Z
%gradients_1/Mean_1_grad/Reshape/shapeConst*
valueB"      *
dtype0

gradients_1/Mean_1_grad/ReshapeReshape-gradients_1/add_grad/tuple/control_dependency%gradients_1/Mean_1_grad/Reshape/shape*
T0*
Tshape0
N
gradients_1/Mean_1_grad/ShapeShapelogistic_loss*
T0*
out_type0

gradients_1/Mean_1_grad/TileTilegradients_1/Mean_1_grad/Reshapegradients_1/Mean_1_grad/Shape*

Tmultiples0*
T0
P
gradients_1/Mean_1_grad/Shape_1Shapelogistic_loss*
T0*
out_type0
H
gradients_1/Mean_1_grad/Shape_2Const*
valueB *
dtype0
K
gradients_1/Mean_1_grad/ConstConst*
valueB: *
dtype0

gradients_1/Mean_1_grad/ProdProdgradients_1/Mean_1_grad/Shape_1gradients_1/Mean_1_grad/Const*

Tidx0*
	keep_dims( *
T0
M
gradients_1/Mean_1_grad/Const_1Const*
valueB: *
dtype0

gradients_1/Mean_1_grad/Prod_1Prodgradients_1/Mean_1_grad/Shape_2gradients_1/Mean_1_grad/Const_1*

Tidx0*
	keep_dims( *
T0
K
!gradients_1/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0
v
gradients_1/Mean_1_grad/MaximumMaximumgradients_1/Mean_1_grad/Prod_1!gradients_1/Mean_1_grad/Maximum/y*
T0
t
 gradients_1/Mean_1_grad/floordivFloorDivgradients_1/Mean_1_grad/Prodgradients_1/Mean_1_grad/Maximum*
T0
n
gradients_1/Mean_1_grad/CastCast gradients_1/Mean_1_grad/floordiv*

SrcT0*
Truncate( *

DstT0
o
gradients_1/Mean_1_grad/truedivRealDivgradients_1/Mean_1_grad/Tilegradients_1/Mean_1_grad/Cast*
T0
Z
%gradients_1/Mean_2_grad/Reshape/shapeConst*
valueB"      *
dtype0

gradients_1/Mean_2_grad/ReshapeReshape/gradients_1/add_grad/tuple/control_dependency_1%gradients_1/Mean_2_grad/Reshape/shape*
T0*
Tshape0
P
gradients_1/Mean_2_grad/ShapeShapelogistic_loss_1*
T0*
out_type0

gradients_1/Mean_2_grad/TileTilegradients_1/Mean_2_grad/Reshapegradients_1/Mean_2_grad/Shape*

Tmultiples0*
T0
R
gradients_1/Mean_2_grad/Shape_1Shapelogistic_loss_1*
T0*
out_type0
H
gradients_1/Mean_2_grad/Shape_2Const*
valueB *
dtype0
K
gradients_1/Mean_2_grad/ConstConst*
valueB: *
dtype0

gradients_1/Mean_2_grad/ProdProdgradients_1/Mean_2_grad/Shape_1gradients_1/Mean_2_grad/Const*

Tidx0*
	keep_dims( *
T0
M
gradients_1/Mean_2_grad/Const_1Const*
valueB: *
dtype0

gradients_1/Mean_2_grad/Prod_1Prodgradients_1/Mean_2_grad/Shape_2gradients_1/Mean_2_grad/Const_1*

Tidx0*
	keep_dims( *
T0
K
!gradients_1/Mean_2_grad/Maximum/yConst*
value	B :*
dtype0
v
gradients_1/Mean_2_grad/MaximumMaximumgradients_1/Mean_2_grad/Prod_1!gradients_1/Mean_2_grad/Maximum/y*
T0
t
 gradients_1/Mean_2_grad/floordivFloorDivgradients_1/Mean_2_grad/Prodgradients_1/Mean_2_grad/Maximum*
T0
n
gradients_1/Mean_2_grad/CastCast gradients_1/Mean_2_grad/floordiv*

SrcT0*
Truncate( *

DstT0
o
gradients_1/Mean_2_grad/truedivRealDivgradients_1/Mean_2_grad/Tilegradients_1/Mean_2_grad/Cast*
T0
Z
%gradients_1/Mean_4_grad/Reshape/shapeConst*
valueB"      *
dtype0

gradients_1/Mean_4_grad/ReshapeReshape/gradients_1/add_1_grad/tuple/control_dependency%gradients_1/Mean_4_grad/Reshape/shape*
T0*
Tshape0
P
gradients_1/Mean_4_grad/ShapeShapelogistic_loss_3*
T0*
out_type0

gradients_1/Mean_4_grad/TileTilegradients_1/Mean_4_grad/Reshapegradients_1/Mean_4_grad/Shape*

Tmultiples0*
T0
R
gradients_1/Mean_4_grad/Shape_1Shapelogistic_loss_3*
T0*
out_type0
H
gradients_1/Mean_4_grad/Shape_2Const*
valueB *
dtype0
K
gradients_1/Mean_4_grad/ConstConst*
valueB: *
dtype0

gradients_1/Mean_4_grad/ProdProdgradients_1/Mean_4_grad/Shape_1gradients_1/Mean_4_grad/Const*

Tidx0*
	keep_dims( *
T0
M
gradients_1/Mean_4_grad/Const_1Const*
valueB: *
dtype0

gradients_1/Mean_4_grad/Prod_1Prodgradients_1/Mean_4_grad/Shape_2gradients_1/Mean_4_grad/Const_1*

Tidx0*
	keep_dims( *
T0
K
!gradients_1/Mean_4_grad/Maximum/yConst*
value	B :*
dtype0
v
gradients_1/Mean_4_grad/MaximumMaximumgradients_1/Mean_4_grad/Prod_1!gradients_1/Mean_4_grad/Maximum/y*
T0
t
 gradients_1/Mean_4_grad/floordivFloorDivgradients_1/Mean_4_grad/Prodgradients_1/Mean_4_grad/Maximum*
T0
n
gradients_1/Mean_4_grad/CastCast gradients_1/Mean_4_grad/floordiv*

SrcT0*
Truncate( *

DstT0
o
gradients_1/Mean_4_grad/truedivRealDivgradients_1/Mean_4_grad/Tilegradients_1/Mean_4_grad/Cast*
T0
Z
%gradients_1/Mean_5_grad/Reshape/shapeConst*
valueB"      *
dtype0

gradients_1/Mean_5_grad/ReshapeReshape1gradients_1/add_1_grad/tuple/control_dependency_1%gradients_1/Mean_5_grad/Reshape/shape*
T0*
Tshape0
P
gradients_1/Mean_5_grad/ShapeShapelogistic_loss_4*
T0*
out_type0

gradients_1/Mean_5_grad/TileTilegradients_1/Mean_5_grad/Reshapegradients_1/Mean_5_grad/Shape*

Tmultiples0*
T0
R
gradients_1/Mean_5_grad/Shape_1Shapelogistic_loss_4*
T0*
out_type0
H
gradients_1/Mean_5_grad/Shape_2Const*
valueB *
dtype0
K
gradients_1/Mean_5_grad/ConstConst*
valueB: *
dtype0

gradients_1/Mean_5_grad/ProdProdgradients_1/Mean_5_grad/Shape_1gradients_1/Mean_5_grad/Const*

Tidx0*
	keep_dims( *
T0
M
gradients_1/Mean_5_grad/Const_1Const*
valueB: *
dtype0

gradients_1/Mean_5_grad/Prod_1Prodgradients_1/Mean_5_grad/Shape_2gradients_1/Mean_5_grad/Const_1*

Tidx0*
	keep_dims( *
T0
K
!gradients_1/Mean_5_grad/Maximum/yConst*
value	B :*
dtype0
v
gradients_1/Mean_5_grad/MaximumMaximumgradients_1/Mean_5_grad/Prod_1!gradients_1/Mean_5_grad/Maximum/y*
T0
t
 gradients_1/Mean_5_grad/floordivFloorDivgradients_1/Mean_5_grad/Prodgradients_1/Mean_5_grad/Maximum*
T0
n
gradients_1/Mean_5_grad/CastCast gradients_1/Mean_5_grad/floordiv*

SrcT0*
Truncate( *

DstT0
o
gradients_1/Mean_5_grad/truedivRealDivgradients_1/Mean_5_grad/Tilegradients_1/Mean_5_grad/Cast*
T0
Y
$gradients_1/logistic_loss_grad/ShapeShapelogistic_loss/sub*
T0*
out_type0
]
&gradients_1/logistic_loss_grad/Shape_1Shapelogistic_loss/Log1p*
T0*
out_type0
¤
4gradients_1/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients_1/logistic_loss_grad/Shape&gradients_1/logistic_loss_grad/Shape_1*
T0
¦
"gradients_1/logistic_loss_grad/SumSumgradients_1/Mean_1_grad/truediv4gradients_1/logistic_loss_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0

&gradients_1/logistic_loss_grad/ReshapeReshape"gradients_1/logistic_loss_grad/Sum$gradients_1/logistic_loss_grad/Shape*
T0*
Tshape0
Ŗ
$gradients_1/logistic_loss_grad/Sum_1Sumgradients_1/Mean_1_grad/truediv6gradients_1/logistic_loss_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0

(gradients_1/logistic_loss_grad/Reshape_1Reshape$gradients_1/logistic_loss_grad/Sum_1&gradients_1/logistic_loss_grad/Shape_1*
T0*
Tshape0

/gradients_1/logistic_loss_grad/tuple/group_depsNoOp'^gradients_1/logistic_loss_grad/Reshape)^gradients_1/logistic_loss_grad/Reshape_1
į
7gradients_1/logistic_loss_grad/tuple/control_dependencyIdentity&gradients_1/logistic_loss_grad/Reshape0^gradients_1/logistic_loss_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_1/logistic_loss_grad/Reshape
ē
9gradients_1/logistic_loss_grad/tuple/control_dependency_1Identity(gradients_1/logistic_loss_grad/Reshape_10^gradients_1/logistic_loss_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/logistic_loss_grad/Reshape_1
]
&gradients_1/logistic_loss_1_grad/ShapeShapelogistic_loss_1/sub*
T0*
out_type0
a
(gradients_1/logistic_loss_1_grad/Shape_1Shapelogistic_loss_1/Log1p*
T0*
out_type0
Ŗ
6gradients_1/logistic_loss_1_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients_1/logistic_loss_1_grad/Shape(gradients_1/logistic_loss_1_grad/Shape_1*
T0
Ŗ
$gradients_1/logistic_loss_1_grad/SumSumgradients_1/Mean_2_grad/truediv6gradients_1/logistic_loss_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0

(gradients_1/logistic_loss_1_grad/ReshapeReshape$gradients_1/logistic_loss_1_grad/Sum&gradients_1/logistic_loss_1_grad/Shape*
T0*
Tshape0
®
&gradients_1/logistic_loss_1_grad/Sum_1Sumgradients_1/Mean_2_grad/truediv8gradients_1/logistic_loss_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0

*gradients_1/logistic_loss_1_grad/Reshape_1Reshape&gradients_1/logistic_loss_1_grad/Sum_1(gradients_1/logistic_loss_1_grad/Shape_1*
T0*
Tshape0

1gradients_1/logistic_loss_1_grad/tuple/group_depsNoOp)^gradients_1/logistic_loss_1_grad/Reshape+^gradients_1/logistic_loss_1_grad/Reshape_1
é
9gradients_1/logistic_loss_1_grad/tuple/control_dependencyIdentity(gradients_1/logistic_loss_1_grad/Reshape2^gradients_1/logistic_loss_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/logistic_loss_1_grad/Reshape
ļ
;gradients_1/logistic_loss_1_grad/tuple/control_dependency_1Identity*gradients_1/logistic_loss_1_grad/Reshape_12^gradients_1/logistic_loss_1_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/logistic_loss_1_grad/Reshape_1
]
&gradients_1/logistic_loss_3_grad/ShapeShapelogistic_loss_3/sub*
T0*
out_type0
a
(gradients_1/logistic_loss_3_grad/Shape_1Shapelogistic_loss_3/Log1p*
T0*
out_type0
Ŗ
6gradients_1/logistic_loss_3_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients_1/logistic_loss_3_grad/Shape(gradients_1/logistic_loss_3_grad/Shape_1*
T0
Ŗ
$gradients_1/logistic_loss_3_grad/SumSumgradients_1/Mean_4_grad/truediv6gradients_1/logistic_loss_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0

(gradients_1/logistic_loss_3_grad/ReshapeReshape$gradients_1/logistic_loss_3_grad/Sum&gradients_1/logistic_loss_3_grad/Shape*
T0*
Tshape0
®
&gradients_1/logistic_loss_3_grad/Sum_1Sumgradients_1/Mean_4_grad/truediv8gradients_1/logistic_loss_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0

*gradients_1/logistic_loss_3_grad/Reshape_1Reshape&gradients_1/logistic_loss_3_grad/Sum_1(gradients_1/logistic_loss_3_grad/Shape_1*
T0*
Tshape0

1gradients_1/logistic_loss_3_grad/tuple/group_depsNoOp)^gradients_1/logistic_loss_3_grad/Reshape+^gradients_1/logistic_loss_3_grad/Reshape_1
é
9gradients_1/logistic_loss_3_grad/tuple/control_dependencyIdentity(gradients_1/logistic_loss_3_grad/Reshape2^gradients_1/logistic_loss_3_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/logistic_loss_3_grad/Reshape
ļ
;gradients_1/logistic_loss_3_grad/tuple/control_dependency_1Identity*gradients_1/logistic_loss_3_grad/Reshape_12^gradients_1/logistic_loss_3_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/logistic_loss_3_grad/Reshape_1
]
&gradients_1/logistic_loss_4_grad/ShapeShapelogistic_loss_4/sub*
T0*
out_type0
a
(gradients_1/logistic_loss_4_grad/Shape_1Shapelogistic_loss_4/Log1p*
T0*
out_type0
Ŗ
6gradients_1/logistic_loss_4_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients_1/logistic_loss_4_grad/Shape(gradients_1/logistic_loss_4_grad/Shape_1*
T0
Ŗ
$gradients_1/logistic_loss_4_grad/SumSumgradients_1/Mean_5_grad/truediv6gradients_1/logistic_loss_4_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0

(gradients_1/logistic_loss_4_grad/ReshapeReshape$gradients_1/logistic_loss_4_grad/Sum&gradients_1/logistic_loss_4_grad/Shape*
T0*
Tshape0
®
&gradients_1/logistic_loss_4_grad/Sum_1Sumgradients_1/Mean_5_grad/truediv8gradients_1/logistic_loss_4_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0

*gradients_1/logistic_loss_4_grad/Reshape_1Reshape&gradients_1/logistic_loss_4_grad/Sum_1(gradients_1/logistic_loss_4_grad/Shape_1*
T0*
Tshape0

1gradients_1/logistic_loss_4_grad/tuple/group_depsNoOp)^gradients_1/logistic_loss_4_grad/Reshape+^gradients_1/logistic_loss_4_grad/Reshape_1
é
9gradients_1/logistic_loss_4_grad/tuple/control_dependencyIdentity(gradients_1/logistic_loss_4_grad/Reshape2^gradients_1/logistic_loss_4_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/logistic_loss_4_grad/Reshape
ļ
;gradients_1/logistic_loss_4_grad/tuple/control_dependency_1Identity*gradients_1/logistic_loss_4_grad/Reshape_12^gradients_1/logistic_loss_4_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/logistic_loss_4_grad/Reshape_1
`
(gradients_1/logistic_loss/sub_grad/ShapeShapelogistic_loss/Select*
T0*
out_type0
_
*gradients_1/logistic_loss/sub_grad/Shape_1Shapelogistic_loss/mul*
T0*
out_type0
°
8gradients_1/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_1/logistic_loss/sub_grad/Shape*gradients_1/logistic_loss/sub_grad/Shape_1*
T0
Ę
&gradients_1/logistic_loss/sub_grad/SumSum7gradients_1/logistic_loss_grad/tuple/control_dependency8gradients_1/logistic_loss/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0

*gradients_1/logistic_loss/sub_grad/ReshapeReshape&gradients_1/logistic_loss/sub_grad/Sum(gradients_1/logistic_loss/sub_grad/Shape*
T0*
Tshape0
o
&gradients_1/logistic_loss/sub_grad/NegNeg7gradients_1/logistic_loss_grad/tuple/control_dependency*
T0
¹
(gradients_1/logistic_loss/sub_grad/Sum_1Sum&gradients_1/logistic_loss/sub_grad/Neg:gradients_1/logistic_loss/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
¤
,gradients_1/logistic_loss/sub_grad/Reshape_1Reshape(gradients_1/logistic_loss/sub_grad/Sum_1*gradients_1/logistic_loss/sub_grad/Shape_1*
T0*
Tshape0

3gradients_1/logistic_loss/sub_grad/tuple/group_depsNoOp+^gradients_1/logistic_loss/sub_grad/Reshape-^gradients_1/logistic_loss/sub_grad/Reshape_1
ń
;gradients_1/logistic_loss/sub_grad/tuple/control_dependencyIdentity*gradients_1/logistic_loss/sub_grad/Reshape4^gradients_1/logistic_loss/sub_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/logistic_loss/sub_grad/Reshape
÷
=gradients_1/logistic_loss/sub_grad/tuple/control_dependency_1Identity,gradients_1/logistic_loss/sub_grad/Reshape_14^gradients_1/logistic_loss/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss/sub_grad/Reshape_1

*gradients_1/logistic_loss/Log1p_grad/add/xConst:^gradients_1/logistic_loss_grad/tuple/control_dependency_1*
valueB
 *  ?*
dtype0
y
(gradients_1/logistic_loss/Log1p_grad/addAddV2*gradients_1/logistic_loss/Log1p_grad/add/xlogistic_loss/Exp*
T0
p
/gradients_1/logistic_loss/Log1p_grad/Reciprocal
Reciprocal(gradients_1/logistic_loss/Log1p_grad/add*
T0
¤
(gradients_1/logistic_loss/Log1p_grad/mulMul9gradients_1/logistic_loss_grad/tuple/control_dependency_1/gradients_1/logistic_loss/Log1p_grad/Reciprocal*
T0
d
*gradients_1/logistic_loss_1/sub_grad/ShapeShapelogistic_loss_1/Select*
T0*
out_type0
c
,gradients_1/logistic_loss_1/sub_grad/Shape_1Shapelogistic_loss_1/mul*
T0*
out_type0
¶
:gradients_1/logistic_loss_1/sub_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/logistic_loss_1/sub_grad/Shape,gradients_1/logistic_loss_1/sub_grad/Shape_1*
T0
Ģ
(gradients_1/logistic_loss_1/sub_grad/SumSum9gradients_1/logistic_loss_1_grad/tuple/control_dependency:gradients_1/logistic_loss_1/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
¤
,gradients_1/logistic_loss_1/sub_grad/ReshapeReshape(gradients_1/logistic_loss_1/sub_grad/Sum*gradients_1/logistic_loss_1/sub_grad/Shape*
T0*
Tshape0
s
(gradients_1/logistic_loss_1/sub_grad/NegNeg9gradients_1/logistic_loss_1_grad/tuple/control_dependency*
T0
æ
*gradients_1/logistic_loss_1/sub_grad/Sum_1Sum(gradients_1/logistic_loss_1/sub_grad/Neg<gradients_1/logistic_loss_1/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ŗ
.gradients_1/logistic_loss_1/sub_grad/Reshape_1Reshape*gradients_1/logistic_loss_1/sub_grad/Sum_1,gradients_1/logistic_loss_1/sub_grad/Shape_1*
T0*
Tshape0

5gradients_1/logistic_loss_1/sub_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss_1/sub_grad/Reshape/^gradients_1/logistic_loss_1/sub_grad/Reshape_1
ł
=gradients_1/logistic_loss_1/sub_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss_1/sub_grad/Reshape6^gradients_1/logistic_loss_1/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss_1/sub_grad/Reshape
’
?gradients_1/logistic_loss_1/sub_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss_1/sub_grad/Reshape_16^gradients_1/logistic_loss_1/sub_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_1/sub_grad/Reshape_1

,gradients_1/logistic_loss_1/Log1p_grad/add/xConst<^gradients_1/logistic_loss_1_grad/tuple/control_dependency_1*
valueB
 *  ?*
dtype0

*gradients_1/logistic_loss_1/Log1p_grad/addAddV2,gradients_1/logistic_loss_1/Log1p_grad/add/xlogistic_loss_1/Exp*
T0
t
1gradients_1/logistic_loss_1/Log1p_grad/Reciprocal
Reciprocal*gradients_1/logistic_loss_1/Log1p_grad/add*
T0
Ŗ
*gradients_1/logistic_loss_1/Log1p_grad/mulMul;gradients_1/logistic_loss_1_grad/tuple/control_dependency_11gradients_1/logistic_loss_1/Log1p_grad/Reciprocal*
T0
d
*gradients_1/logistic_loss_3/sub_grad/ShapeShapelogistic_loss_3/Select*
T0*
out_type0
c
,gradients_1/logistic_loss_3/sub_grad/Shape_1Shapelogistic_loss_3/mul*
T0*
out_type0
¶
:gradients_1/logistic_loss_3/sub_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/logistic_loss_3/sub_grad/Shape,gradients_1/logistic_loss_3/sub_grad/Shape_1*
T0
Ģ
(gradients_1/logistic_loss_3/sub_grad/SumSum9gradients_1/logistic_loss_3_grad/tuple/control_dependency:gradients_1/logistic_loss_3/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
¤
,gradients_1/logistic_loss_3/sub_grad/ReshapeReshape(gradients_1/logistic_loss_3/sub_grad/Sum*gradients_1/logistic_loss_3/sub_grad/Shape*
T0*
Tshape0
s
(gradients_1/logistic_loss_3/sub_grad/NegNeg9gradients_1/logistic_loss_3_grad/tuple/control_dependency*
T0
æ
*gradients_1/logistic_loss_3/sub_grad/Sum_1Sum(gradients_1/logistic_loss_3/sub_grad/Neg<gradients_1/logistic_loss_3/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ŗ
.gradients_1/logistic_loss_3/sub_grad/Reshape_1Reshape*gradients_1/logistic_loss_3/sub_grad/Sum_1,gradients_1/logistic_loss_3/sub_grad/Shape_1*
T0*
Tshape0

5gradients_1/logistic_loss_3/sub_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss_3/sub_grad/Reshape/^gradients_1/logistic_loss_3/sub_grad/Reshape_1
ł
=gradients_1/logistic_loss_3/sub_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss_3/sub_grad/Reshape6^gradients_1/logistic_loss_3/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss_3/sub_grad/Reshape
’
?gradients_1/logistic_loss_3/sub_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss_3/sub_grad/Reshape_16^gradients_1/logistic_loss_3/sub_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_3/sub_grad/Reshape_1

,gradients_1/logistic_loss_3/Log1p_grad/add/xConst<^gradients_1/logistic_loss_3_grad/tuple/control_dependency_1*
valueB
 *  ?*
dtype0

*gradients_1/logistic_loss_3/Log1p_grad/addAddV2,gradients_1/logistic_loss_3/Log1p_grad/add/xlogistic_loss_3/Exp*
T0
t
1gradients_1/logistic_loss_3/Log1p_grad/Reciprocal
Reciprocal*gradients_1/logistic_loss_3/Log1p_grad/add*
T0
Ŗ
*gradients_1/logistic_loss_3/Log1p_grad/mulMul;gradients_1/logistic_loss_3_grad/tuple/control_dependency_11gradients_1/logistic_loss_3/Log1p_grad/Reciprocal*
T0
d
*gradients_1/logistic_loss_4/sub_grad/ShapeShapelogistic_loss_4/Select*
T0*
out_type0
c
,gradients_1/logistic_loss_4/sub_grad/Shape_1Shapelogistic_loss_4/mul*
T0*
out_type0
¶
:gradients_1/logistic_loss_4/sub_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/logistic_loss_4/sub_grad/Shape,gradients_1/logistic_loss_4/sub_grad/Shape_1*
T0
Ģ
(gradients_1/logistic_loss_4/sub_grad/SumSum9gradients_1/logistic_loss_4_grad/tuple/control_dependency:gradients_1/logistic_loss_4/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
¤
,gradients_1/logistic_loss_4/sub_grad/ReshapeReshape(gradients_1/logistic_loss_4/sub_grad/Sum*gradients_1/logistic_loss_4/sub_grad/Shape*
T0*
Tshape0
s
(gradients_1/logistic_loss_4/sub_grad/NegNeg9gradients_1/logistic_loss_4_grad/tuple/control_dependency*
T0
æ
*gradients_1/logistic_loss_4/sub_grad/Sum_1Sum(gradients_1/logistic_loss_4/sub_grad/Neg<gradients_1/logistic_loss_4/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ŗ
.gradients_1/logistic_loss_4/sub_grad/Reshape_1Reshape*gradients_1/logistic_loss_4/sub_grad/Sum_1,gradients_1/logistic_loss_4/sub_grad/Shape_1*
T0*
Tshape0

5gradients_1/logistic_loss_4/sub_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss_4/sub_grad/Reshape/^gradients_1/logistic_loss_4/sub_grad/Reshape_1
ł
=gradients_1/logistic_loss_4/sub_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss_4/sub_grad/Reshape6^gradients_1/logistic_loss_4/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss_4/sub_grad/Reshape
’
?gradients_1/logistic_loss_4/sub_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss_4/sub_grad/Reshape_16^gradients_1/logistic_loss_4/sub_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_4/sub_grad/Reshape_1

,gradients_1/logistic_loss_4/Log1p_grad/add/xConst<^gradients_1/logistic_loss_4_grad/tuple/control_dependency_1*
valueB
 *  ?*
dtype0

*gradients_1/logistic_loss_4/Log1p_grad/addAddV2,gradients_1/logistic_loss_4/Log1p_grad/add/xlogistic_loss_4/Exp*
T0
t
1gradients_1/logistic_loss_4/Log1p_grad/Reciprocal
Reciprocal*gradients_1/logistic_loss_4/Log1p_grad/add*
T0
Ŗ
*gradients_1/logistic_loss_4/Log1p_grad/mulMul;gradients_1/logistic_loss_4_grad/tuple/control_dependency_11gradients_1/logistic_loss_4/Log1p_grad/Reciprocal*
T0
[
0gradients_1/logistic_loss/Select_grad/zeros_like	ZerosLikediscriminator/add_2*
T0
Ź
,gradients_1/logistic_loss/Select_grad/SelectSelectlogistic_loss/GreaterEqual;gradients_1/logistic_loss/sub_grad/tuple/control_dependency0gradients_1/logistic_loss/Select_grad/zeros_like*
T0
Ģ
.gradients_1/logistic_loss/Select_grad/Select_1Selectlogistic_loss/GreaterEqual0gradients_1/logistic_loss/Select_grad/zeros_like;gradients_1/logistic_loss/sub_grad/tuple/control_dependency*
T0

6gradients_1/logistic_loss/Select_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss/Select_grad/Select/^gradients_1/logistic_loss/Select_grad/Select_1
ū
>gradients_1/logistic_loss/Select_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss/Select_grad/Select7^gradients_1/logistic_loss/Select_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss/Select_grad/Select

@gradients_1/logistic_loss/Select_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss/Select_grad/Select_17^gradients_1/logistic_loss/Select_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss/Select_grad/Select_1
_
(gradients_1/logistic_loss/mul_grad/ShapeShapediscriminator/add_2*
T0*
out_type0
W
*gradients_1/logistic_loss/mul_grad/Shape_1Shape	ones_like*
T0*
out_type0
°
8gradients_1/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_1/logistic_loss/mul_grad/Shape*gradients_1/logistic_loss/mul_grad/Shape_1*
T0

&gradients_1/logistic_loss/mul_grad/MulMul=gradients_1/logistic_loss/sub_grad/tuple/control_dependency_1	ones_like*
T0
µ
&gradients_1/logistic_loss/mul_grad/SumSum&gradients_1/logistic_loss/mul_grad/Mul8gradients_1/logistic_loss/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0

*gradients_1/logistic_loss/mul_grad/ReshapeReshape&gradients_1/logistic_loss/mul_grad/Sum(gradients_1/logistic_loss/mul_grad/Shape*
T0*
Tshape0

(gradients_1/logistic_loss/mul_grad/Mul_1Muldiscriminator/add_2=gradients_1/logistic_loss/sub_grad/tuple/control_dependency_1*
T0
»
(gradients_1/logistic_loss/mul_grad/Sum_1Sum(gradients_1/logistic_loss/mul_grad/Mul_1:gradients_1/logistic_loss/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
¤
,gradients_1/logistic_loss/mul_grad/Reshape_1Reshape(gradients_1/logistic_loss/mul_grad/Sum_1*gradients_1/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0

3gradients_1/logistic_loss/mul_grad/tuple/group_depsNoOp+^gradients_1/logistic_loss/mul_grad/Reshape-^gradients_1/logistic_loss/mul_grad/Reshape_1
ń
;gradients_1/logistic_loss/mul_grad/tuple/control_dependencyIdentity*gradients_1/logistic_loss/mul_grad/Reshape4^gradients_1/logistic_loss/mul_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/logistic_loss/mul_grad/Reshape
÷
=gradients_1/logistic_loss/mul_grad/tuple/control_dependency_1Identity,gradients_1/logistic_loss/mul_grad/Reshape_14^gradients_1/logistic_loss/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss/mul_grad/Reshape_1
s
&gradients_1/logistic_loss/Exp_grad/mulMul(gradients_1/logistic_loss/Log1p_grad/mullogistic_loss/Exp*
T0
_
2gradients_1/logistic_loss_1/Select_grad/zeros_like	ZerosLikediscriminator_1/add_2*
T0
Ņ
.gradients_1/logistic_loss_1/Select_grad/SelectSelectlogistic_loss_1/GreaterEqual=gradients_1/logistic_loss_1/sub_grad/tuple/control_dependency2gradients_1/logistic_loss_1/Select_grad/zeros_like*
T0
Ō
0gradients_1/logistic_loss_1/Select_grad/Select_1Selectlogistic_loss_1/GreaterEqual2gradients_1/logistic_loss_1/Select_grad/zeros_like=gradients_1/logistic_loss_1/sub_grad/tuple/control_dependency*
T0
¤
8gradients_1/logistic_loss_1/Select_grad/tuple/group_depsNoOp/^gradients_1/logistic_loss_1/Select_grad/Select1^gradients_1/logistic_loss_1/Select_grad/Select_1

@gradients_1/logistic_loss_1/Select_grad/tuple/control_dependencyIdentity.gradients_1/logistic_loss_1/Select_grad/Select9^gradients_1/logistic_loss_1/Select_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_1/Select_grad/Select

Bgradients_1/logistic_loss_1/Select_grad/tuple/control_dependency_1Identity0gradients_1/logistic_loss_1/Select_grad/Select_19^gradients_1/logistic_loss_1/Select_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/logistic_loss_1/Select_grad/Select_1
c
*gradients_1/logistic_loss_1/mul_grad/ShapeShapediscriminator_1/add_2*
T0*
out_type0
Z
,gradients_1/logistic_loss_1/mul_grad/Shape_1Shape
zeros_like*
T0*
out_type0
¶
:gradients_1/logistic_loss_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/logistic_loss_1/mul_grad/Shape,gradients_1/logistic_loss_1/mul_grad/Shape_1*
T0

(gradients_1/logistic_loss_1/mul_grad/MulMul?gradients_1/logistic_loss_1/sub_grad/tuple/control_dependency_1
zeros_like*
T0
»
(gradients_1/logistic_loss_1/mul_grad/SumSum(gradients_1/logistic_loss_1/mul_grad/Mul:gradients_1/logistic_loss_1/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
¤
,gradients_1/logistic_loss_1/mul_grad/ReshapeReshape(gradients_1/logistic_loss_1/mul_grad/Sum*gradients_1/logistic_loss_1/mul_grad/Shape*
T0*
Tshape0

*gradients_1/logistic_loss_1/mul_grad/Mul_1Muldiscriminator_1/add_2?gradients_1/logistic_loss_1/sub_grad/tuple/control_dependency_1*
T0
Į
*gradients_1/logistic_loss_1/mul_grad/Sum_1Sum*gradients_1/logistic_loss_1/mul_grad/Mul_1<gradients_1/logistic_loss_1/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ŗ
.gradients_1/logistic_loss_1/mul_grad/Reshape_1Reshape*gradients_1/logistic_loss_1/mul_grad/Sum_1,gradients_1/logistic_loss_1/mul_grad/Shape_1*
T0*
Tshape0

5gradients_1/logistic_loss_1/mul_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss_1/mul_grad/Reshape/^gradients_1/logistic_loss_1/mul_grad/Reshape_1
ł
=gradients_1/logistic_loss_1/mul_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss_1/mul_grad/Reshape6^gradients_1/logistic_loss_1/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss_1/mul_grad/Reshape
’
?gradients_1/logistic_loss_1/mul_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss_1/mul_grad/Reshape_16^gradients_1/logistic_loss_1/mul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_1/mul_grad/Reshape_1
y
(gradients_1/logistic_loss_1/Exp_grad/mulMul*gradients_1/logistic_loss_1/Log1p_grad/mullogistic_loss_1/Exp*
T0
a
2gradients_1/logistic_loss_3/Select_grad/zeros_like	ZerosLikediscriminator_cat/add_2*
T0
Ņ
.gradients_1/logistic_loss_3/Select_grad/SelectSelectlogistic_loss_3/GreaterEqual=gradients_1/logistic_loss_3/sub_grad/tuple/control_dependency2gradients_1/logistic_loss_3/Select_grad/zeros_like*
T0
Ō
0gradients_1/logistic_loss_3/Select_grad/Select_1Selectlogistic_loss_3/GreaterEqual2gradients_1/logistic_loss_3/Select_grad/zeros_like=gradients_1/logistic_loss_3/sub_grad/tuple/control_dependency*
T0
¤
8gradients_1/logistic_loss_3/Select_grad/tuple/group_depsNoOp/^gradients_1/logistic_loss_3/Select_grad/Select1^gradients_1/logistic_loss_3/Select_grad/Select_1

@gradients_1/logistic_loss_3/Select_grad/tuple/control_dependencyIdentity.gradients_1/logistic_loss_3/Select_grad/Select9^gradients_1/logistic_loss_3/Select_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_3/Select_grad/Select

Bgradients_1/logistic_loss_3/Select_grad/tuple/control_dependency_1Identity0gradients_1/logistic_loss_3/Select_grad/Select_19^gradients_1/logistic_loss_3/Select_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/logistic_loss_3/Select_grad/Select_1
e
*gradients_1/logistic_loss_3/mul_grad/ShapeShapediscriminator_cat/add_2*
T0*
out_type0
[
,gradients_1/logistic_loss_3/mul_grad/Shape_1Shapeones_like_2*
T0*
out_type0
¶
:gradients_1/logistic_loss_3/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/logistic_loss_3/mul_grad/Shape,gradients_1/logistic_loss_3/mul_grad/Shape_1*
T0

(gradients_1/logistic_loss_3/mul_grad/MulMul?gradients_1/logistic_loss_3/sub_grad/tuple/control_dependency_1ones_like_2*
T0
»
(gradients_1/logistic_loss_3/mul_grad/SumSum(gradients_1/logistic_loss_3/mul_grad/Mul:gradients_1/logistic_loss_3/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
¤
,gradients_1/logistic_loss_3/mul_grad/ReshapeReshape(gradients_1/logistic_loss_3/mul_grad/Sum*gradients_1/logistic_loss_3/mul_grad/Shape*
T0*
Tshape0

*gradients_1/logistic_loss_3/mul_grad/Mul_1Muldiscriminator_cat/add_2?gradients_1/logistic_loss_3/sub_grad/tuple/control_dependency_1*
T0
Į
*gradients_1/logistic_loss_3/mul_grad/Sum_1Sum*gradients_1/logistic_loss_3/mul_grad/Mul_1<gradients_1/logistic_loss_3/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ŗ
.gradients_1/logistic_loss_3/mul_grad/Reshape_1Reshape*gradients_1/logistic_loss_3/mul_grad/Sum_1,gradients_1/logistic_loss_3/mul_grad/Shape_1*
T0*
Tshape0

5gradients_1/logistic_loss_3/mul_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss_3/mul_grad/Reshape/^gradients_1/logistic_loss_3/mul_grad/Reshape_1
ł
=gradients_1/logistic_loss_3/mul_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss_3/mul_grad/Reshape6^gradients_1/logistic_loss_3/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss_3/mul_grad/Reshape
’
?gradients_1/logistic_loss_3/mul_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss_3/mul_grad/Reshape_16^gradients_1/logistic_loss_3/mul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_3/mul_grad/Reshape_1
y
(gradients_1/logistic_loss_3/Exp_grad/mulMul*gradients_1/logistic_loss_3/Log1p_grad/mullogistic_loss_3/Exp*
T0
c
2gradients_1/logistic_loss_4/Select_grad/zeros_like	ZerosLikediscriminator_cat_1/add_2*
T0
Ņ
.gradients_1/logistic_loss_4/Select_grad/SelectSelectlogistic_loss_4/GreaterEqual=gradients_1/logistic_loss_4/sub_grad/tuple/control_dependency2gradients_1/logistic_loss_4/Select_grad/zeros_like*
T0
Ō
0gradients_1/logistic_loss_4/Select_grad/Select_1Selectlogistic_loss_4/GreaterEqual2gradients_1/logistic_loss_4/Select_grad/zeros_like=gradients_1/logistic_loss_4/sub_grad/tuple/control_dependency*
T0
¤
8gradients_1/logistic_loss_4/Select_grad/tuple/group_depsNoOp/^gradients_1/logistic_loss_4/Select_grad/Select1^gradients_1/logistic_loss_4/Select_grad/Select_1

@gradients_1/logistic_loss_4/Select_grad/tuple/control_dependencyIdentity.gradients_1/logistic_loss_4/Select_grad/Select9^gradients_1/logistic_loss_4/Select_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_4/Select_grad/Select

Bgradients_1/logistic_loss_4/Select_grad/tuple/control_dependency_1Identity0gradients_1/logistic_loss_4/Select_grad/Select_19^gradients_1/logistic_loss_4/Select_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/logistic_loss_4/Select_grad/Select_1
g
*gradients_1/logistic_loss_4/mul_grad/ShapeShapediscriminator_cat_1/add_2*
T0*
out_type0
\
,gradients_1/logistic_loss_4/mul_grad/Shape_1Shapezeros_like_1*
T0*
out_type0
¶
:gradients_1/logistic_loss_4/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/logistic_loss_4/mul_grad/Shape,gradients_1/logistic_loss_4/mul_grad/Shape_1*
T0

(gradients_1/logistic_loss_4/mul_grad/MulMul?gradients_1/logistic_loss_4/sub_grad/tuple/control_dependency_1zeros_like_1*
T0
»
(gradients_1/logistic_loss_4/mul_grad/SumSum(gradients_1/logistic_loss_4/mul_grad/Mul:gradients_1/logistic_loss_4/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
¤
,gradients_1/logistic_loss_4/mul_grad/ReshapeReshape(gradients_1/logistic_loss_4/mul_grad/Sum*gradients_1/logistic_loss_4/mul_grad/Shape*
T0*
Tshape0

*gradients_1/logistic_loss_4/mul_grad/Mul_1Muldiscriminator_cat_1/add_2?gradients_1/logistic_loss_4/sub_grad/tuple/control_dependency_1*
T0
Į
*gradients_1/logistic_loss_4/mul_grad/Sum_1Sum*gradients_1/logistic_loss_4/mul_grad/Mul_1<gradients_1/logistic_loss_4/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ŗ
.gradients_1/logistic_loss_4/mul_grad/Reshape_1Reshape*gradients_1/logistic_loss_4/mul_grad/Sum_1,gradients_1/logistic_loss_4/mul_grad/Shape_1*
T0*
Tshape0

5gradients_1/logistic_loss_4/mul_grad/tuple/group_depsNoOp-^gradients_1/logistic_loss_4/mul_grad/Reshape/^gradients_1/logistic_loss_4/mul_grad/Reshape_1
ł
=gradients_1/logistic_loss_4/mul_grad/tuple/control_dependencyIdentity,gradients_1/logistic_loss_4/mul_grad/Reshape6^gradients_1/logistic_loss_4/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/logistic_loss_4/mul_grad/Reshape
’
?gradients_1/logistic_loss_4/mul_grad/tuple/control_dependency_1Identity.gradients_1/logistic_loss_4/mul_grad/Reshape_16^gradients_1/logistic_loss_4/mul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss_4/mul_grad/Reshape_1
y
(gradients_1/logistic_loss_4/Exp_grad/mulMul*gradients_1/logistic_loss_4/Log1p_grad/mullogistic_loss_4/Exp*
T0
[
2gradients_1/logistic_loss/Select_1_grad/zeros_like	ZerosLikelogistic_loss/Neg*
T0
¹
.gradients_1/logistic_loss/Select_1_grad/SelectSelectlogistic_loss/GreaterEqual&gradients_1/logistic_loss/Exp_grad/mul2gradients_1/logistic_loss/Select_1_grad/zeros_like*
T0
»
0gradients_1/logistic_loss/Select_1_grad/Select_1Selectlogistic_loss/GreaterEqual2gradients_1/logistic_loss/Select_1_grad/zeros_like&gradients_1/logistic_loss/Exp_grad/mul*
T0
¤
8gradients_1/logistic_loss/Select_1_grad/tuple/group_depsNoOp/^gradients_1/logistic_loss/Select_1_grad/Select1^gradients_1/logistic_loss/Select_1_grad/Select_1

@gradients_1/logistic_loss/Select_1_grad/tuple/control_dependencyIdentity.gradients_1/logistic_loss/Select_1_grad/Select9^gradients_1/logistic_loss/Select_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/logistic_loss/Select_1_grad/Select

Bgradients_1/logistic_loss/Select_1_grad/tuple/control_dependency_1Identity0gradients_1/logistic_loss/Select_1_grad/Select_19^gradients_1/logistic_loss/Select_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/logistic_loss/Select_1_grad/Select_1
_
4gradients_1/logistic_loss_1/Select_1_grad/zeros_like	ZerosLikelogistic_loss_1/Neg*
T0
Į
0gradients_1/logistic_loss_1/Select_1_grad/SelectSelectlogistic_loss_1/GreaterEqual(gradients_1/logistic_loss_1/Exp_grad/mul4gradients_1/logistic_loss_1/Select_1_grad/zeros_like*
T0
Ć
2gradients_1/logistic_loss_1/Select_1_grad/Select_1Selectlogistic_loss_1/GreaterEqual4gradients_1/logistic_loss_1/Select_1_grad/zeros_like(gradients_1/logistic_loss_1/Exp_grad/mul*
T0
Ŗ
:gradients_1/logistic_loss_1/Select_1_grad/tuple/group_depsNoOp1^gradients_1/logistic_loss_1/Select_1_grad/Select3^gradients_1/logistic_loss_1/Select_1_grad/Select_1

Bgradients_1/logistic_loss_1/Select_1_grad/tuple/control_dependencyIdentity0gradients_1/logistic_loss_1/Select_1_grad/Select;^gradients_1/logistic_loss_1/Select_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/logistic_loss_1/Select_1_grad/Select

Dgradients_1/logistic_loss_1/Select_1_grad/tuple/control_dependency_1Identity2gradients_1/logistic_loss_1/Select_1_grad/Select_1;^gradients_1/logistic_loss_1/Select_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/logistic_loss_1/Select_1_grad/Select_1
_
4gradients_1/logistic_loss_3/Select_1_grad/zeros_like	ZerosLikelogistic_loss_3/Neg*
T0
Į
0gradients_1/logistic_loss_3/Select_1_grad/SelectSelectlogistic_loss_3/GreaterEqual(gradients_1/logistic_loss_3/Exp_grad/mul4gradients_1/logistic_loss_3/Select_1_grad/zeros_like*
T0
Ć
2gradients_1/logistic_loss_3/Select_1_grad/Select_1Selectlogistic_loss_3/GreaterEqual4gradients_1/logistic_loss_3/Select_1_grad/zeros_like(gradients_1/logistic_loss_3/Exp_grad/mul*
T0
Ŗ
:gradients_1/logistic_loss_3/Select_1_grad/tuple/group_depsNoOp1^gradients_1/logistic_loss_3/Select_1_grad/Select3^gradients_1/logistic_loss_3/Select_1_grad/Select_1

Bgradients_1/logistic_loss_3/Select_1_grad/tuple/control_dependencyIdentity0gradients_1/logistic_loss_3/Select_1_grad/Select;^gradients_1/logistic_loss_3/Select_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/logistic_loss_3/Select_1_grad/Select

Dgradients_1/logistic_loss_3/Select_1_grad/tuple/control_dependency_1Identity2gradients_1/logistic_loss_3/Select_1_grad/Select_1;^gradients_1/logistic_loss_3/Select_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/logistic_loss_3/Select_1_grad/Select_1
_
4gradients_1/logistic_loss_4/Select_1_grad/zeros_like	ZerosLikelogistic_loss_4/Neg*
T0
Į
0gradients_1/logistic_loss_4/Select_1_grad/SelectSelectlogistic_loss_4/GreaterEqual(gradients_1/logistic_loss_4/Exp_grad/mul4gradients_1/logistic_loss_4/Select_1_grad/zeros_like*
T0
Ć
2gradients_1/logistic_loss_4/Select_1_grad/Select_1Selectlogistic_loss_4/GreaterEqual4gradients_1/logistic_loss_4/Select_1_grad/zeros_like(gradients_1/logistic_loss_4/Exp_grad/mul*
T0
Ŗ
:gradients_1/logistic_loss_4/Select_1_grad/tuple/group_depsNoOp1^gradients_1/logistic_loss_4/Select_1_grad/Select3^gradients_1/logistic_loss_4/Select_1_grad/Select_1

Bgradients_1/logistic_loss_4/Select_1_grad/tuple/control_dependencyIdentity0gradients_1/logistic_loss_4/Select_1_grad/Select;^gradients_1/logistic_loss_4/Select_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/logistic_loss_4/Select_1_grad/Select

Dgradients_1/logistic_loss_4/Select_1_grad/tuple/control_dependency_1Identity2gradients_1/logistic_loss_4/Select_1_grad/Select_1;^gradients_1/logistic_loss_4/Select_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/logistic_loss_4/Select_1_grad/Select_1
x
&gradients_1/logistic_loss/Neg_grad/NegNeg@gradients_1/logistic_loss/Select_1_grad/tuple/control_dependency*
T0
|
(gradients_1/logistic_loss_1/Neg_grad/NegNegBgradients_1/logistic_loss_1/Select_1_grad/tuple/control_dependency*
T0
|
(gradients_1/logistic_loss_3/Neg_grad/NegNegBgradients_1/logistic_loss_3/Select_1_grad/tuple/control_dependency*
T0
|
(gradients_1/logistic_loss_4/Neg_grad/NegNegBgradients_1/logistic_loss_4/Select_1_grad/tuple/control_dependency*
T0
Ō
gradients_1/AddNAddN>gradients_1/logistic_loss/Select_grad/tuple/control_dependency;gradients_1/logistic_loss/mul_grad/tuple/control_dependencyBgradients_1/logistic_loss/Select_1_grad/tuple/control_dependency_1&gradients_1/logistic_loss/Neg_grad/Neg*
T0*?
_class5
31loc:@gradients_1/logistic_loss/Select_grad/Select*
N
d
*gradients_1/discriminator/add_2_grad/ShapeShapediscriminator/MatMul_2*
T0*
out_type0
e
,gradients_1/discriminator/add_2_grad/Shape_1Shapediscriminator/bo/read*
T0*
out_type0
¶
:gradients_1/discriminator/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/discriminator/add_2_grad/Shape,gradients_1/discriminator/add_2_grad/Shape_1*
T0
£
(gradients_1/discriminator/add_2_grad/SumSumgradients_1/AddN:gradients_1/discriminator/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
¤
,gradients_1/discriminator/add_2_grad/ReshapeReshape(gradients_1/discriminator/add_2_grad/Sum*gradients_1/discriminator/add_2_grad/Shape*
T0*
Tshape0
§
*gradients_1/discriminator/add_2_grad/Sum_1Sumgradients_1/AddN<gradients_1/discriminator/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ŗ
.gradients_1/discriminator/add_2_grad/Reshape_1Reshape*gradients_1/discriminator/add_2_grad/Sum_1,gradients_1/discriminator/add_2_grad/Shape_1*
T0*
Tshape0

5gradients_1/discriminator/add_2_grad/tuple/group_depsNoOp-^gradients_1/discriminator/add_2_grad/Reshape/^gradients_1/discriminator/add_2_grad/Reshape_1
ł
=gradients_1/discriminator/add_2_grad/tuple/control_dependencyIdentity,gradients_1/discriminator/add_2_grad/Reshape6^gradients_1/discriminator/add_2_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/discriminator/add_2_grad/Reshape
’
?gradients_1/discriminator/add_2_grad/tuple/control_dependency_1Identity.gradients_1/discriminator/add_2_grad/Reshape_16^gradients_1/discriminator/add_2_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/discriminator/add_2_grad/Reshape_1
ą
gradients_1/AddN_1AddN@gradients_1/logistic_loss_1/Select_grad/tuple/control_dependency=gradients_1/logistic_loss_1/mul_grad/tuple/control_dependencyDgradients_1/logistic_loss_1/Select_1_grad/tuple/control_dependency_1(gradients_1/logistic_loss_1/Neg_grad/Neg*
T0*A
_class7
53loc:@gradients_1/logistic_loss_1/Select_grad/Select*
N
h
,gradients_1/discriminator_1/add_2_grad/ShapeShapediscriminator_1/MatMul_2*
T0*
out_type0
g
.gradients_1/discriminator_1/add_2_grad/Shape_1Shapediscriminator/bo/read*
T0*
out_type0
¼
<gradients_1/discriminator_1/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients_1/discriminator_1/add_2_grad/Shape.gradients_1/discriminator_1/add_2_grad/Shape_1*
T0
©
*gradients_1/discriminator_1/add_2_grad/SumSumgradients_1/AddN_1<gradients_1/discriminator_1/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
Ŗ
.gradients_1/discriminator_1/add_2_grad/ReshapeReshape*gradients_1/discriminator_1/add_2_grad/Sum,gradients_1/discriminator_1/add_2_grad/Shape*
T0*
Tshape0
­
,gradients_1/discriminator_1/add_2_grad/Sum_1Sumgradients_1/AddN_1>gradients_1/discriminator_1/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
°
0gradients_1/discriminator_1/add_2_grad/Reshape_1Reshape,gradients_1/discriminator_1/add_2_grad/Sum_1.gradients_1/discriminator_1/add_2_grad/Shape_1*
T0*
Tshape0
£
7gradients_1/discriminator_1/add_2_grad/tuple/group_depsNoOp/^gradients_1/discriminator_1/add_2_grad/Reshape1^gradients_1/discriminator_1/add_2_grad/Reshape_1

?gradients_1/discriminator_1/add_2_grad/tuple/control_dependencyIdentity.gradients_1/discriminator_1/add_2_grad/Reshape8^gradients_1/discriminator_1/add_2_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/discriminator_1/add_2_grad/Reshape

Agradients_1/discriminator_1/add_2_grad/tuple/control_dependency_1Identity0gradients_1/discriminator_1/add_2_grad/Reshape_18^gradients_1/discriminator_1/add_2_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/discriminator_1/add_2_grad/Reshape_1
ą
gradients_1/AddN_2AddN@gradients_1/logistic_loss_3/Select_grad/tuple/control_dependency=gradients_1/logistic_loss_3/mul_grad/tuple/control_dependencyDgradients_1/logistic_loss_3/Select_1_grad/tuple/control_dependency_1(gradients_1/logistic_loss_3/Neg_grad/Neg*
T0*A
_class7
53loc:@gradients_1/logistic_loss_3/Select_grad/Select*
N
l
.gradients_1/discriminator_cat/add_2_grad/ShapeShapediscriminator_cat/MatMul_2*
T0*
out_type0
m
0gradients_1/discriminator_cat/add_2_grad/Shape_1Shapediscriminator_cat/bo/read*
T0*
out_type0
Ā
>gradients_1/discriminator_cat/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients_1/discriminator_cat/add_2_grad/Shape0gradients_1/discriminator_cat/add_2_grad/Shape_1*
T0
­
,gradients_1/discriminator_cat/add_2_grad/SumSumgradients_1/AddN_2>gradients_1/discriminator_cat/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
°
0gradients_1/discriminator_cat/add_2_grad/ReshapeReshape,gradients_1/discriminator_cat/add_2_grad/Sum.gradients_1/discriminator_cat/add_2_grad/Shape*
T0*
Tshape0
±
.gradients_1/discriminator_cat/add_2_grad/Sum_1Sumgradients_1/AddN_2@gradients_1/discriminator_cat/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
¶
2gradients_1/discriminator_cat/add_2_grad/Reshape_1Reshape.gradients_1/discriminator_cat/add_2_grad/Sum_10gradients_1/discriminator_cat/add_2_grad/Shape_1*
T0*
Tshape0
©
9gradients_1/discriminator_cat/add_2_grad/tuple/group_depsNoOp1^gradients_1/discriminator_cat/add_2_grad/Reshape3^gradients_1/discriminator_cat/add_2_grad/Reshape_1

Agradients_1/discriminator_cat/add_2_grad/tuple/control_dependencyIdentity0gradients_1/discriminator_cat/add_2_grad/Reshape:^gradients_1/discriminator_cat/add_2_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/discriminator_cat/add_2_grad/Reshape

Cgradients_1/discriminator_cat/add_2_grad/tuple/control_dependency_1Identity2gradients_1/discriminator_cat/add_2_grad/Reshape_1:^gradients_1/discriminator_cat/add_2_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/discriminator_cat/add_2_grad/Reshape_1
ą
gradients_1/AddN_3AddN@gradients_1/logistic_loss_4/Select_grad/tuple/control_dependency=gradients_1/logistic_loss_4/mul_grad/tuple/control_dependencyDgradients_1/logistic_loss_4/Select_1_grad/tuple/control_dependency_1(gradients_1/logistic_loss_4/Neg_grad/Neg*
T0*A
_class7
53loc:@gradients_1/logistic_loss_4/Select_grad/Select*
N
p
0gradients_1/discriminator_cat_1/add_2_grad/ShapeShapediscriminator_cat_1/MatMul_2*
T0*
out_type0
o
2gradients_1/discriminator_cat_1/add_2_grad/Shape_1Shapediscriminator_cat/bo/read*
T0*
out_type0
Č
@gradients_1/discriminator_cat_1/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs0gradients_1/discriminator_cat_1/add_2_grad/Shape2gradients_1/discriminator_cat_1/add_2_grad/Shape_1*
T0
±
.gradients_1/discriminator_cat_1/add_2_grad/SumSumgradients_1/AddN_3@gradients_1/discriminator_cat_1/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
¶
2gradients_1/discriminator_cat_1/add_2_grad/ReshapeReshape.gradients_1/discriminator_cat_1/add_2_grad/Sum0gradients_1/discriminator_cat_1/add_2_grad/Shape*
T0*
Tshape0
µ
0gradients_1/discriminator_cat_1/add_2_grad/Sum_1Sumgradients_1/AddN_3Bgradients_1/discriminator_cat_1/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
¼
4gradients_1/discriminator_cat_1/add_2_grad/Reshape_1Reshape0gradients_1/discriminator_cat_1/add_2_grad/Sum_12gradients_1/discriminator_cat_1/add_2_grad/Shape_1*
T0*
Tshape0
Æ
;gradients_1/discriminator_cat_1/add_2_grad/tuple/group_depsNoOp3^gradients_1/discriminator_cat_1/add_2_grad/Reshape5^gradients_1/discriminator_cat_1/add_2_grad/Reshape_1

Cgradients_1/discriminator_cat_1/add_2_grad/tuple/control_dependencyIdentity2gradients_1/discriminator_cat_1/add_2_grad/Reshape<^gradients_1/discriminator_cat_1/add_2_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/discriminator_cat_1/add_2_grad/Reshape

Egradients_1/discriminator_cat_1/add_2_grad/tuple/control_dependency_1Identity4gradients_1/discriminator_cat_1/add_2_grad/Reshape_1<^gradients_1/discriminator_cat_1/add_2_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/discriminator_cat_1/add_2_grad/Reshape_1
½
.gradients_1/discriminator/MatMul_2_grad/MatMulMatMul=gradients_1/discriminator/add_2_grad/tuple/control_dependencydiscriminator/wo/read*
transpose_b(*
T0*
transpose_a( 
Ē
0gradients_1/discriminator/MatMul_2_grad/MatMul_1MatMuldiscriminator/dropout_1/mul_1=gradients_1/discriminator/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
¤
8gradients_1/discriminator/MatMul_2_grad/tuple/group_depsNoOp/^gradients_1/discriminator/MatMul_2_grad/MatMul1^gradients_1/discriminator/MatMul_2_grad/MatMul_1

@gradients_1/discriminator/MatMul_2_grad/tuple/control_dependencyIdentity.gradients_1/discriminator/MatMul_2_grad/MatMul9^gradients_1/discriminator/MatMul_2_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/discriminator/MatMul_2_grad/MatMul

Bgradients_1/discriminator/MatMul_2_grad/tuple/control_dependency_1Identity0gradients_1/discriminator/MatMul_2_grad/MatMul_19^gradients_1/discriminator/MatMul_2_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/discriminator/MatMul_2_grad/MatMul_1
Į
0gradients_1/discriminator_1/MatMul_2_grad/MatMulMatMul?gradients_1/discriminator_1/add_2_grad/tuple/control_dependencydiscriminator/wo/read*
transpose_b(*
T0*
transpose_a( 
Ķ
2gradients_1/discriminator_1/MatMul_2_grad/MatMul_1MatMuldiscriminator_1/dropout_1/mul_1?gradients_1/discriminator_1/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
Ŗ
:gradients_1/discriminator_1/MatMul_2_grad/tuple/group_depsNoOp1^gradients_1/discriminator_1/MatMul_2_grad/MatMul3^gradients_1/discriminator_1/MatMul_2_grad/MatMul_1

Bgradients_1/discriminator_1/MatMul_2_grad/tuple/control_dependencyIdentity0gradients_1/discriminator_1/MatMul_2_grad/MatMul;^gradients_1/discriminator_1/MatMul_2_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/discriminator_1/MatMul_2_grad/MatMul

Dgradients_1/discriminator_1/MatMul_2_grad/tuple/control_dependency_1Identity2gradients_1/discriminator_1/MatMul_2_grad/MatMul_1;^gradients_1/discriminator_1/MatMul_2_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/discriminator_1/MatMul_2_grad/MatMul_1
ó
gradients_1/AddN_4AddN?gradients_1/discriminator/add_2_grad/tuple/control_dependency_1Agradients_1/discriminator_1/add_2_grad/tuple/control_dependency_1*
T0*A
_class7
53loc:@gradients_1/discriminator/add_2_grad/Reshape_1*
N
É
2gradients_1/discriminator_cat/MatMul_2_grad/MatMulMatMulAgradients_1/discriminator_cat/add_2_grad/tuple/control_dependencydiscriminator_cat/wo/read*
transpose_b(*
T0*
transpose_a( 
Ó
4gradients_1/discriminator_cat/MatMul_2_grad/MatMul_1MatMul!discriminator_cat/dropout_1/mul_1Agradients_1/discriminator_cat/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
°
<gradients_1/discriminator_cat/MatMul_2_grad/tuple/group_depsNoOp3^gradients_1/discriminator_cat/MatMul_2_grad/MatMul5^gradients_1/discriminator_cat/MatMul_2_grad/MatMul_1

Dgradients_1/discriminator_cat/MatMul_2_grad/tuple/control_dependencyIdentity2gradients_1/discriminator_cat/MatMul_2_grad/MatMul=^gradients_1/discriminator_cat/MatMul_2_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/discriminator_cat/MatMul_2_grad/MatMul

Fgradients_1/discriminator_cat/MatMul_2_grad/tuple/control_dependency_1Identity4gradients_1/discriminator_cat/MatMul_2_grad/MatMul_1=^gradients_1/discriminator_cat/MatMul_2_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/discriminator_cat/MatMul_2_grad/MatMul_1
Ķ
4gradients_1/discriminator_cat_1/MatMul_2_grad/MatMulMatMulCgradients_1/discriminator_cat_1/add_2_grad/tuple/control_dependencydiscriminator_cat/wo/read*
transpose_b(*
T0*
transpose_a( 
Ł
6gradients_1/discriminator_cat_1/MatMul_2_grad/MatMul_1MatMul#discriminator_cat_1/dropout_1/mul_1Cgradients_1/discriminator_cat_1/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
¶
>gradients_1/discriminator_cat_1/MatMul_2_grad/tuple/group_depsNoOp5^gradients_1/discriminator_cat_1/MatMul_2_grad/MatMul7^gradients_1/discriminator_cat_1/MatMul_2_grad/MatMul_1

Fgradients_1/discriminator_cat_1/MatMul_2_grad/tuple/control_dependencyIdentity4gradients_1/discriminator_cat_1/MatMul_2_grad/MatMul?^gradients_1/discriminator_cat_1/MatMul_2_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/discriminator_cat_1/MatMul_2_grad/MatMul
”
Hgradients_1/discriminator_cat_1/MatMul_2_grad/tuple/control_dependency_1Identity6gradients_1/discriminator_cat_1/MatMul_2_grad/MatMul_1?^gradients_1/discriminator_cat_1/MatMul_2_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients_1/discriminator_cat_1/MatMul_2_grad/MatMul_1
’
gradients_1/AddN_5AddNCgradients_1/discriminator_cat/add_2_grad/tuple/control_dependency_1Egradients_1/discriminator_cat_1/add_2_grad/tuple/control_dependency_1*
T0*E
_class;
97loc:@gradients_1/discriminator_cat/add_2_grad/Reshape_1*
N
s
4gradients_1/discriminator/dropout_1/mul_1_grad/ShapeShapediscriminator/dropout_1/mul*
T0*
out_type0
v
6gradients_1/discriminator/dropout_1/mul_1_grad/Shape_1Shapediscriminator/dropout_1/Cast*
T0*
out_type0
Ō
Dgradients_1/discriminator/dropout_1/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients_1/discriminator/dropout_1/mul_1_grad/Shape6gradients_1/discriminator/dropout_1/mul_1_grad/Shape_1*
T0
¢
2gradients_1/discriminator/dropout_1/mul_1_grad/MulMul@gradients_1/discriminator/MatMul_2_grad/tuple/control_dependencydiscriminator/dropout_1/Cast*
T0
Ł
2gradients_1/discriminator/dropout_1/mul_1_grad/SumSum2gradients_1/discriminator/dropout_1/mul_1_grad/MulDgradients_1/discriminator/dropout_1/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
Ā
6gradients_1/discriminator/dropout_1/mul_1_grad/ReshapeReshape2gradients_1/discriminator/dropout_1/mul_1_grad/Sum4gradients_1/discriminator/dropout_1/mul_1_grad/Shape*
T0*
Tshape0
£
4gradients_1/discriminator/dropout_1/mul_1_grad/Mul_1Muldiscriminator/dropout_1/mul@gradients_1/discriminator/MatMul_2_grad/tuple/control_dependency*
T0
ß
4gradients_1/discriminator/dropout_1/mul_1_grad/Sum_1Sum4gradients_1/discriminator/dropout_1/mul_1_grad/Mul_1Fgradients_1/discriminator/dropout_1/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Č
8gradients_1/discriminator/dropout_1/mul_1_grad/Reshape_1Reshape4gradients_1/discriminator/dropout_1/mul_1_grad/Sum_16gradients_1/discriminator/dropout_1/mul_1_grad/Shape_1*
T0*
Tshape0
»
?gradients_1/discriminator/dropout_1/mul_1_grad/tuple/group_depsNoOp7^gradients_1/discriminator/dropout_1/mul_1_grad/Reshape9^gradients_1/discriminator/dropout_1/mul_1_grad/Reshape_1
”
Ggradients_1/discriminator/dropout_1/mul_1_grad/tuple/control_dependencyIdentity6gradients_1/discriminator/dropout_1/mul_1_grad/Reshape@^gradients_1/discriminator/dropout_1/mul_1_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients_1/discriminator/dropout_1/mul_1_grad/Reshape
§
Igradients_1/discriminator/dropout_1/mul_1_grad/tuple/control_dependency_1Identity8gradients_1/discriminator/dropout_1/mul_1_grad/Reshape_1@^gradients_1/discriminator/dropout_1/mul_1_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/discriminator/dropout_1/mul_1_grad/Reshape_1
w
6gradients_1/discriminator_1/dropout_1/mul_1_grad/ShapeShapediscriminator_1/dropout_1/mul*
T0*
out_type0
z
8gradients_1/discriminator_1/dropout_1/mul_1_grad/Shape_1Shapediscriminator_1/dropout_1/Cast*
T0*
out_type0
Ś
Fgradients_1/discriminator_1/dropout_1/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_1/discriminator_1/dropout_1/mul_1_grad/Shape8gradients_1/discriminator_1/dropout_1/mul_1_grad/Shape_1*
T0
Ø
4gradients_1/discriminator_1/dropout_1/mul_1_grad/MulMulBgradients_1/discriminator_1/MatMul_2_grad/tuple/control_dependencydiscriminator_1/dropout_1/Cast*
T0
ß
4gradients_1/discriminator_1/dropout_1/mul_1_grad/SumSum4gradients_1/discriminator_1/dropout_1/mul_1_grad/MulFgradients_1/discriminator_1/dropout_1/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
Č
8gradients_1/discriminator_1/dropout_1/mul_1_grad/ReshapeReshape4gradients_1/discriminator_1/dropout_1/mul_1_grad/Sum6gradients_1/discriminator_1/dropout_1/mul_1_grad/Shape*
T0*
Tshape0
©
6gradients_1/discriminator_1/dropout_1/mul_1_grad/Mul_1Muldiscriminator_1/dropout_1/mulBgradients_1/discriminator_1/MatMul_2_grad/tuple/control_dependency*
T0
å
6gradients_1/discriminator_1/dropout_1/mul_1_grad/Sum_1Sum6gradients_1/discriminator_1/dropout_1/mul_1_grad/Mul_1Hgradients_1/discriminator_1/dropout_1/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ī
:gradients_1/discriminator_1/dropout_1/mul_1_grad/Reshape_1Reshape6gradients_1/discriminator_1/dropout_1/mul_1_grad/Sum_18gradients_1/discriminator_1/dropout_1/mul_1_grad/Shape_1*
T0*
Tshape0
Į
Agradients_1/discriminator_1/dropout_1/mul_1_grad/tuple/group_depsNoOp9^gradients_1/discriminator_1/dropout_1/mul_1_grad/Reshape;^gradients_1/discriminator_1/dropout_1/mul_1_grad/Reshape_1
©
Igradients_1/discriminator_1/dropout_1/mul_1_grad/tuple/control_dependencyIdentity8gradients_1/discriminator_1/dropout_1/mul_1_grad/ReshapeB^gradients_1/discriminator_1/dropout_1/mul_1_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/discriminator_1/dropout_1/mul_1_grad/Reshape
Æ
Kgradients_1/discriminator_1/dropout_1/mul_1_grad/tuple/control_dependency_1Identity:gradients_1/discriminator_1/dropout_1/mul_1_grad/Reshape_1B^gradients_1/discriminator_1/dropout_1/mul_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/discriminator_1/dropout_1/mul_1_grad/Reshape_1
ū
gradients_1/AddN_6AddNBgradients_1/discriminator/MatMul_2_grad/tuple/control_dependency_1Dgradients_1/discriminator_1/MatMul_2_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients_1/discriminator/MatMul_2_grad/MatMul_1*
N
{
8gradients_1/discriminator_cat/dropout_1/mul_1_grad/ShapeShapediscriminator_cat/dropout_1/mul*
T0*
out_type0
~
:gradients_1/discriminator_cat/dropout_1/mul_1_grad/Shape_1Shape discriminator_cat/dropout_1/Cast*
T0*
out_type0
ą
Hgradients_1/discriminator_cat/dropout_1/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients_1/discriminator_cat/dropout_1/mul_1_grad/Shape:gradients_1/discriminator_cat/dropout_1/mul_1_grad/Shape_1*
T0
®
6gradients_1/discriminator_cat/dropout_1/mul_1_grad/MulMulDgradients_1/discriminator_cat/MatMul_2_grad/tuple/control_dependency discriminator_cat/dropout_1/Cast*
T0
å
6gradients_1/discriminator_cat/dropout_1/mul_1_grad/SumSum6gradients_1/discriminator_cat/dropout_1/mul_1_grad/MulHgradients_1/discriminator_cat/dropout_1/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
Ī
:gradients_1/discriminator_cat/dropout_1/mul_1_grad/ReshapeReshape6gradients_1/discriminator_cat/dropout_1/mul_1_grad/Sum8gradients_1/discriminator_cat/dropout_1/mul_1_grad/Shape*
T0*
Tshape0
Æ
8gradients_1/discriminator_cat/dropout_1/mul_1_grad/Mul_1Muldiscriminator_cat/dropout_1/mulDgradients_1/discriminator_cat/MatMul_2_grad/tuple/control_dependency*
T0
ė
8gradients_1/discriminator_cat/dropout_1/mul_1_grad/Sum_1Sum8gradients_1/discriminator_cat/dropout_1/mul_1_grad/Mul_1Jgradients_1/discriminator_cat/dropout_1/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ō
<gradients_1/discriminator_cat/dropout_1/mul_1_grad/Reshape_1Reshape8gradients_1/discriminator_cat/dropout_1/mul_1_grad/Sum_1:gradients_1/discriminator_cat/dropout_1/mul_1_grad/Shape_1*
T0*
Tshape0
Ē
Cgradients_1/discriminator_cat/dropout_1/mul_1_grad/tuple/group_depsNoOp;^gradients_1/discriminator_cat/dropout_1/mul_1_grad/Reshape=^gradients_1/discriminator_cat/dropout_1/mul_1_grad/Reshape_1
±
Kgradients_1/discriminator_cat/dropout_1/mul_1_grad/tuple/control_dependencyIdentity:gradients_1/discriminator_cat/dropout_1/mul_1_grad/ReshapeD^gradients_1/discriminator_cat/dropout_1/mul_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/discriminator_cat/dropout_1/mul_1_grad/Reshape
·
Mgradients_1/discriminator_cat/dropout_1/mul_1_grad/tuple/control_dependency_1Identity<gradients_1/discriminator_cat/dropout_1/mul_1_grad/Reshape_1D^gradients_1/discriminator_cat/dropout_1/mul_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_1/discriminator_cat/dropout_1/mul_1_grad/Reshape_1

:gradients_1/discriminator_cat_1/dropout_1/mul_1_grad/ShapeShape!discriminator_cat_1/dropout_1/mul*
T0*
out_type0

<gradients_1/discriminator_cat_1/dropout_1/mul_1_grad/Shape_1Shape"discriminator_cat_1/dropout_1/Cast*
T0*
out_type0
ę
Jgradients_1/discriminator_cat_1/dropout_1/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients_1/discriminator_cat_1/dropout_1/mul_1_grad/Shape<gradients_1/discriminator_cat_1/dropout_1/mul_1_grad/Shape_1*
T0
“
8gradients_1/discriminator_cat_1/dropout_1/mul_1_grad/MulMulFgradients_1/discriminator_cat_1/MatMul_2_grad/tuple/control_dependency"discriminator_cat_1/dropout_1/Cast*
T0
ė
8gradients_1/discriminator_cat_1/dropout_1/mul_1_grad/SumSum8gradients_1/discriminator_cat_1/dropout_1/mul_1_grad/MulJgradients_1/discriminator_cat_1/dropout_1/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
Ō
<gradients_1/discriminator_cat_1/dropout_1/mul_1_grad/ReshapeReshape8gradients_1/discriminator_cat_1/dropout_1/mul_1_grad/Sum:gradients_1/discriminator_cat_1/dropout_1/mul_1_grad/Shape*
T0*
Tshape0
µ
:gradients_1/discriminator_cat_1/dropout_1/mul_1_grad/Mul_1Mul!discriminator_cat_1/dropout_1/mulFgradients_1/discriminator_cat_1/MatMul_2_grad/tuple/control_dependency*
T0
ń
:gradients_1/discriminator_cat_1/dropout_1/mul_1_grad/Sum_1Sum:gradients_1/discriminator_cat_1/dropout_1/mul_1_grad/Mul_1Lgradients_1/discriminator_cat_1/dropout_1/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ś
>gradients_1/discriminator_cat_1/dropout_1/mul_1_grad/Reshape_1Reshape:gradients_1/discriminator_cat_1/dropout_1/mul_1_grad/Sum_1<gradients_1/discriminator_cat_1/dropout_1/mul_1_grad/Shape_1*
T0*
Tshape0
Ķ
Egradients_1/discriminator_cat_1/dropout_1/mul_1_grad/tuple/group_depsNoOp=^gradients_1/discriminator_cat_1/dropout_1/mul_1_grad/Reshape?^gradients_1/discriminator_cat_1/dropout_1/mul_1_grad/Reshape_1
¹
Mgradients_1/discriminator_cat_1/dropout_1/mul_1_grad/tuple/control_dependencyIdentity<gradients_1/discriminator_cat_1/dropout_1/mul_1_grad/ReshapeF^gradients_1/discriminator_cat_1/dropout_1/mul_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_1/discriminator_cat_1/dropout_1/mul_1_grad/Reshape
æ
Ogradients_1/discriminator_cat_1/dropout_1/mul_1_grad/tuple/control_dependency_1Identity>gradients_1/discriminator_cat_1/dropout_1/mul_1_grad/Reshape_1F^gradients_1/discriminator_cat_1/dropout_1/mul_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients_1/discriminator_cat_1/dropout_1/mul_1_grad/Reshape_1

gradients_1/AddN_7AddNFgradients_1/discriminator_cat/MatMul_2_grad/tuple/control_dependency_1Hgradients_1/discriminator_cat_1/MatMul_2_grad/tuple/control_dependency_1*
T0*G
_class=
;9loc:@gradients_1/discriminator_cat/MatMul_2_grad/MatMul_1*
N
j
2gradients_1/discriminator/dropout_1/mul_grad/ShapeShapediscriminator/Relu_1*
T0*
out_type0
w
4gradients_1/discriminator/dropout_1/mul_grad/Shape_1Shapediscriminator/dropout_1/truediv*
T0*
out_type0
Ī
Bgradients_1/discriminator/dropout_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients_1/discriminator/dropout_1/mul_grad/Shape4gradients_1/discriminator/dropout_1/mul_grad/Shape_1*
T0
Ŗ
0gradients_1/discriminator/dropout_1/mul_grad/MulMulGgradients_1/discriminator/dropout_1/mul_1_grad/tuple/control_dependencydiscriminator/dropout_1/truediv*
T0
Ó
0gradients_1/discriminator/dropout_1/mul_grad/SumSum0gradients_1/discriminator/dropout_1/mul_grad/MulBgradients_1/discriminator/dropout_1/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
¼
4gradients_1/discriminator/dropout_1/mul_grad/ReshapeReshape0gradients_1/discriminator/dropout_1/mul_grad/Sum2gradients_1/discriminator/dropout_1/mul_grad/Shape*
T0*
Tshape0
”
2gradients_1/discriminator/dropout_1/mul_grad/Mul_1Muldiscriminator/Relu_1Ggradients_1/discriminator/dropout_1/mul_1_grad/tuple/control_dependency*
T0
Ł
2gradients_1/discriminator/dropout_1/mul_grad/Sum_1Sum2gradients_1/discriminator/dropout_1/mul_grad/Mul_1Dgradients_1/discriminator/dropout_1/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ā
6gradients_1/discriminator/dropout_1/mul_grad/Reshape_1Reshape2gradients_1/discriminator/dropout_1/mul_grad/Sum_14gradients_1/discriminator/dropout_1/mul_grad/Shape_1*
T0*
Tshape0
µ
=gradients_1/discriminator/dropout_1/mul_grad/tuple/group_depsNoOp5^gradients_1/discriminator/dropout_1/mul_grad/Reshape7^gradients_1/discriminator/dropout_1/mul_grad/Reshape_1

Egradients_1/discriminator/dropout_1/mul_grad/tuple/control_dependencyIdentity4gradients_1/discriminator/dropout_1/mul_grad/Reshape>^gradients_1/discriminator/dropout_1/mul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/discriminator/dropout_1/mul_grad/Reshape

Ggradients_1/discriminator/dropout_1/mul_grad/tuple/control_dependency_1Identity6gradients_1/discriminator/dropout_1/mul_grad/Reshape_1>^gradients_1/discriminator/dropout_1/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients_1/discriminator/dropout_1/mul_grad/Reshape_1
n
4gradients_1/discriminator_1/dropout_1/mul_grad/ShapeShapediscriminator_1/Relu_1*
T0*
out_type0
{
6gradients_1/discriminator_1/dropout_1/mul_grad/Shape_1Shape!discriminator_1/dropout_1/truediv*
T0*
out_type0
Ō
Dgradients_1/discriminator_1/dropout_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients_1/discriminator_1/dropout_1/mul_grad/Shape6gradients_1/discriminator_1/dropout_1/mul_grad/Shape_1*
T0
°
2gradients_1/discriminator_1/dropout_1/mul_grad/MulMulIgradients_1/discriminator_1/dropout_1/mul_1_grad/tuple/control_dependency!discriminator_1/dropout_1/truediv*
T0
Ł
2gradients_1/discriminator_1/dropout_1/mul_grad/SumSum2gradients_1/discriminator_1/dropout_1/mul_grad/MulDgradients_1/discriminator_1/dropout_1/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
Ā
6gradients_1/discriminator_1/dropout_1/mul_grad/ReshapeReshape2gradients_1/discriminator_1/dropout_1/mul_grad/Sum4gradients_1/discriminator_1/dropout_1/mul_grad/Shape*
T0*
Tshape0
§
4gradients_1/discriminator_1/dropout_1/mul_grad/Mul_1Muldiscriminator_1/Relu_1Igradients_1/discriminator_1/dropout_1/mul_1_grad/tuple/control_dependency*
T0
ß
4gradients_1/discriminator_1/dropout_1/mul_grad/Sum_1Sum4gradients_1/discriminator_1/dropout_1/mul_grad/Mul_1Fgradients_1/discriminator_1/dropout_1/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Č
8gradients_1/discriminator_1/dropout_1/mul_grad/Reshape_1Reshape4gradients_1/discriminator_1/dropout_1/mul_grad/Sum_16gradients_1/discriminator_1/dropout_1/mul_grad/Shape_1*
T0*
Tshape0
»
?gradients_1/discriminator_1/dropout_1/mul_grad/tuple/group_depsNoOp7^gradients_1/discriminator_1/dropout_1/mul_grad/Reshape9^gradients_1/discriminator_1/dropout_1/mul_grad/Reshape_1
”
Ggradients_1/discriminator_1/dropout_1/mul_grad/tuple/control_dependencyIdentity6gradients_1/discriminator_1/dropout_1/mul_grad/Reshape@^gradients_1/discriminator_1/dropout_1/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients_1/discriminator_1/dropout_1/mul_grad/Reshape
§
Igradients_1/discriminator_1/dropout_1/mul_grad/tuple/control_dependency_1Identity8gradients_1/discriminator_1/dropout_1/mul_grad/Reshape_1@^gradients_1/discriminator_1/dropout_1/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/discriminator_1/dropout_1/mul_grad/Reshape_1
r
6gradients_1/discriminator_cat/dropout_1/mul_grad/ShapeShapediscriminator_cat/Relu_1*
T0*
out_type0

8gradients_1/discriminator_cat/dropout_1/mul_grad/Shape_1Shape#discriminator_cat/dropout_1/truediv*
T0*
out_type0
Ś
Fgradients_1/discriminator_cat/dropout_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_1/discriminator_cat/dropout_1/mul_grad/Shape8gradients_1/discriminator_cat/dropout_1/mul_grad/Shape_1*
T0
¶
4gradients_1/discriminator_cat/dropout_1/mul_grad/MulMulKgradients_1/discriminator_cat/dropout_1/mul_1_grad/tuple/control_dependency#discriminator_cat/dropout_1/truediv*
T0
ß
4gradients_1/discriminator_cat/dropout_1/mul_grad/SumSum4gradients_1/discriminator_cat/dropout_1/mul_grad/MulFgradients_1/discriminator_cat/dropout_1/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
Č
8gradients_1/discriminator_cat/dropout_1/mul_grad/ReshapeReshape4gradients_1/discriminator_cat/dropout_1/mul_grad/Sum6gradients_1/discriminator_cat/dropout_1/mul_grad/Shape*
T0*
Tshape0
­
6gradients_1/discriminator_cat/dropout_1/mul_grad/Mul_1Muldiscriminator_cat/Relu_1Kgradients_1/discriminator_cat/dropout_1/mul_1_grad/tuple/control_dependency*
T0
å
6gradients_1/discriminator_cat/dropout_1/mul_grad/Sum_1Sum6gradients_1/discriminator_cat/dropout_1/mul_grad/Mul_1Hgradients_1/discriminator_cat/dropout_1/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ī
:gradients_1/discriminator_cat/dropout_1/mul_grad/Reshape_1Reshape6gradients_1/discriminator_cat/dropout_1/mul_grad/Sum_18gradients_1/discriminator_cat/dropout_1/mul_grad/Shape_1*
T0*
Tshape0
Į
Agradients_1/discriminator_cat/dropout_1/mul_grad/tuple/group_depsNoOp9^gradients_1/discriminator_cat/dropout_1/mul_grad/Reshape;^gradients_1/discriminator_cat/dropout_1/mul_grad/Reshape_1
©
Igradients_1/discriminator_cat/dropout_1/mul_grad/tuple/control_dependencyIdentity8gradients_1/discriminator_cat/dropout_1/mul_grad/ReshapeB^gradients_1/discriminator_cat/dropout_1/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/discriminator_cat/dropout_1/mul_grad/Reshape
Æ
Kgradients_1/discriminator_cat/dropout_1/mul_grad/tuple/control_dependency_1Identity:gradients_1/discriminator_cat/dropout_1/mul_grad/Reshape_1B^gradients_1/discriminator_cat/dropout_1/mul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/discriminator_cat/dropout_1/mul_grad/Reshape_1
v
8gradients_1/discriminator_cat_1/dropout_1/mul_grad/ShapeShapediscriminator_cat_1/Relu_1*
T0*
out_type0

:gradients_1/discriminator_cat_1/dropout_1/mul_grad/Shape_1Shape%discriminator_cat_1/dropout_1/truediv*
T0*
out_type0
ą
Hgradients_1/discriminator_cat_1/dropout_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients_1/discriminator_cat_1/dropout_1/mul_grad/Shape:gradients_1/discriminator_cat_1/dropout_1/mul_grad/Shape_1*
T0
¼
6gradients_1/discriminator_cat_1/dropout_1/mul_grad/MulMulMgradients_1/discriminator_cat_1/dropout_1/mul_1_grad/tuple/control_dependency%discriminator_cat_1/dropout_1/truediv*
T0
å
6gradients_1/discriminator_cat_1/dropout_1/mul_grad/SumSum6gradients_1/discriminator_cat_1/dropout_1/mul_grad/MulHgradients_1/discriminator_cat_1/dropout_1/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
Ī
:gradients_1/discriminator_cat_1/dropout_1/mul_grad/ReshapeReshape6gradients_1/discriminator_cat_1/dropout_1/mul_grad/Sum8gradients_1/discriminator_cat_1/dropout_1/mul_grad/Shape*
T0*
Tshape0
³
8gradients_1/discriminator_cat_1/dropout_1/mul_grad/Mul_1Muldiscriminator_cat_1/Relu_1Mgradients_1/discriminator_cat_1/dropout_1/mul_1_grad/tuple/control_dependency*
T0
ė
8gradients_1/discriminator_cat_1/dropout_1/mul_grad/Sum_1Sum8gradients_1/discriminator_cat_1/dropout_1/mul_grad/Mul_1Jgradients_1/discriminator_cat_1/dropout_1/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ō
<gradients_1/discriminator_cat_1/dropout_1/mul_grad/Reshape_1Reshape8gradients_1/discriminator_cat_1/dropout_1/mul_grad/Sum_1:gradients_1/discriminator_cat_1/dropout_1/mul_grad/Shape_1*
T0*
Tshape0
Ē
Cgradients_1/discriminator_cat_1/dropout_1/mul_grad/tuple/group_depsNoOp;^gradients_1/discriminator_cat_1/dropout_1/mul_grad/Reshape=^gradients_1/discriminator_cat_1/dropout_1/mul_grad/Reshape_1
±
Kgradients_1/discriminator_cat_1/dropout_1/mul_grad/tuple/control_dependencyIdentity:gradients_1/discriminator_cat_1/dropout_1/mul_grad/ReshapeD^gradients_1/discriminator_cat_1/dropout_1/mul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/discriminator_cat_1/dropout_1/mul_grad/Reshape
·
Mgradients_1/discriminator_cat_1/dropout_1/mul_grad/tuple/control_dependency_1Identity<gradients_1/discriminator_cat_1/dropout_1/mul_grad/Reshape_1D^gradients_1/discriminator_cat_1/dropout_1/mul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_1/discriminator_cat_1/dropout_1/mul_grad/Reshape_1
 
.gradients_1/discriminator/Relu_1_grad/ReluGradReluGradEgradients_1/discriminator/dropout_1/mul_grad/tuple/control_dependencydiscriminator/Relu_1*
T0
¦
0gradients_1/discriminator_1/Relu_1_grad/ReluGradReluGradGgradients_1/discriminator_1/dropout_1/mul_grad/tuple/control_dependencydiscriminator_1/Relu_1*
T0
¬
2gradients_1/discriminator_cat/Relu_1_grad/ReluGradReluGradIgradients_1/discriminator_cat/dropout_1/mul_grad/tuple/control_dependencydiscriminator_cat/Relu_1*
T0
²
4gradients_1/discriminator_cat_1/Relu_1_grad/ReluGradReluGradKgradients_1/discriminator_cat_1/dropout_1/mul_grad/tuple/control_dependencydiscriminator_cat_1/Relu_1*
T0
d
*gradients_1/discriminator/add_1_grad/ShapeShapediscriminator/MatMul_1*
T0*
out_type0
e
,gradients_1/discriminator/add_1_grad/Shape_1Shapediscriminator/b1/read*
T0*
out_type0
¶
:gradients_1/discriminator/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/discriminator/add_1_grad/Shape,gradients_1/discriminator/add_1_grad/Shape_1*
T0
Į
(gradients_1/discriminator/add_1_grad/SumSum.gradients_1/discriminator/Relu_1_grad/ReluGrad:gradients_1/discriminator/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
¤
,gradients_1/discriminator/add_1_grad/ReshapeReshape(gradients_1/discriminator/add_1_grad/Sum*gradients_1/discriminator/add_1_grad/Shape*
T0*
Tshape0
Å
*gradients_1/discriminator/add_1_grad/Sum_1Sum.gradients_1/discriminator/Relu_1_grad/ReluGrad<gradients_1/discriminator/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ŗ
.gradients_1/discriminator/add_1_grad/Reshape_1Reshape*gradients_1/discriminator/add_1_grad/Sum_1,gradients_1/discriminator/add_1_grad/Shape_1*
T0*
Tshape0

5gradients_1/discriminator/add_1_grad/tuple/group_depsNoOp-^gradients_1/discriminator/add_1_grad/Reshape/^gradients_1/discriminator/add_1_grad/Reshape_1
ł
=gradients_1/discriminator/add_1_grad/tuple/control_dependencyIdentity,gradients_1/discriminator/add_1_grad/Reshape6^gradients_1/discriminator/add_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/discriminator/add_1_grad/Reshape
’
?gradients_1/discriminator/add_1_grad/tuple/control_dependency_1Identity.gradients_1/discriminator/add_1_grad/Reshape_16^gradients_1/discriminator/add_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/discriminator/add_1_grad/Reshape_1
h
,gradients_1/discriminator_1/add_1_grad/ShapeShapediscriminator_1/MatMul_1*
T0*
out_type0
g
.gradients_1/discriminator_1/add_1_grad/Shape_1Shapediscriminator/b1/read*
T0*
out_type0
¼
<gradients_1/discriminator_1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients_1/discriminator_1/add_1_grad/Shape.gradients_1/discriminator_1/add_1_grad/Shape_1*
T0
Ē
*gradients_1/discriminator_1/add_1_grad/SumSum0gradients_1/discriminator_1/Relu_1_grad/ReluGrad<gradients_1/discriminator_1/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
Ŗ
.gradients_1/discriminator_1/add_1_grad/ReshapeReshape*gradients_1/discriminator_1/add_1_grad/Sum,gradients_1/discriminator_1/add_1_grad/Shape*
T0*
Tshape0
Ė
,gradients_1/discriminator_1/add_1_grad/Sum_1Sum0gradients_1/discriminator_1/Relu_1_grad/ReluGrad>gradients_1/discriminator_1/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
°
0gradients_1/discriminator_1/add_1_grad/Reshape_1Reshape,gradients_1/discriminator_1/add_1_grad/Sum_1.gradients_1/discriminator_1/add_1_grad/Shape_1*
T0*
Tshape0
£
7gradients_1/discriminator_1/add_1_grad/tuple/group_depsNoOp/^gradients_1/discriminator_1/add_1_grad/Reshape1^gradients_1/discriminator_1/add_1_grad/Reshape_1

?gradients_1/discriminator_1/add_1_grad/tuple/control_dependencyIdentity.gradients_1/discriminator_1/add_1_grad/Reshape8^gradients_1/discriminator_1/add_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/discriminator_1/add_1_grad/Reshape

Agradients_1/discriminator_1/add_1_grad/tuple/control_dependency_1Identity0gradients_1/discriminator_1/add_1_grad/Reshape_18^gradients_1/discriminator_1/add_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/discriminator_1/add_1_grad/Reshape_1
l
.gradients_1/discriminator_cat/add_1_grad/ShapeShapediscriminator_cat/MatMul_1*
T0*
out_type0
m
0gradients_1/discriminator_cat/add_1_grad/Shape_1Shapediscriminator_cat/b1/read*
T0*
out_type0
Ā
>gradients_1/discriminator_cat/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients_1/discriminator_cat/add_1_grad/Shape0gradients_1/discriminator_cat/add_1_grad/Shape_1*
T0
Ķ
,gradients_1/discriminator_cat/add_1_grad/SumSum2gradients_1/discriminator_cat/Relu_1_grad/ReluGrad>gradients_1/discriminator_cat/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
°
0gradients_1/discriminator_cat/add_1_grad/ReshapeReshape,gradients_1/discriminator_cat/add_1_grad/Sum.gradients_1/discriminator_cat/add_1_grad/Shape*
T0*
Tshape0
Ń
.gradients_1/discriminator_cat/add_1_grad/Sum_1Sum2gradients_1/discriminator_cat/Relu_1_grad/ReluGrad@gradients_1/discriminator_cat/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
¶
2gradients_1/discriminator_cat/add_1_grad/Reshape_1Reshape.gradients_1/discriminator_cat/add_1_grad/Sum_10gradients_1/discriminator_cat/add_1_grad/Shape_1*
T0*
Tshape0
©
9gradients_1/discriminator_cat/add_1_grad/tuple/group_depsNoOp1^gradients_1/discriminator_cat/add_1_grad/Reshape3^gradients_1/discriminator_cat/add_1_grad/Reshape_1

Agradients_1/discriminator_cat/add_1_grad/tuple/control_dependencyIdentity0gradients_1/discriminator_cat/add_1_grad/Reshape:^gradients_1/discriminator_cat/add_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/discriminator_cat/add_1_grad/Reshape

Cgradients_1/discriminator_cat/add_1_grad/tuple/control_dependency_1Identity2gradients_1/discriminator_cat/add_1_grad/Reshape_1:^gradients_1/discriminator_cat/add_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/discriminator_cat/add_1_grad/Reshape_1
p
0gradients_1/discriminator_cat_1/add_1_grad/ShapeShapediscriminator_cat_1/MatMul_1*
T0*
out_type0
o
2gradients_1/discriminator_cat_1/add_1_grad/Shape_1Shapediscriminator_cat/b1/read*
T0*
out_type0
Č
@gradients_1/discriminator_cat_1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs0gradients_1/discriminator_cat_1/add_1_grad/Shape2gradients_1/discriminator_cat_1/add_1_grad/Shape_1*
T0
Ó
.gradients_1/discriminator_cat_1/add_1_grad/SumSum4gradients_1/discriminator_cat_1/Relu_1_grad/ReluGrad@gradients_1/discriminator_cat_1/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
¶
2gradients_1/discriminator_cat_1/add_1_grad/ReshapeReshape.gradients_1/discriminator_cat_1/add_1_grad/Sum0gradients_1/discriminator_cat_1/add_1_grad/Shape*
T0*
Tshape0
×
0gradients_1/discriminator_cat_1/add_1_grad/Sum_1Sum4gradients_1/discriminator_cat_1/Relu_1_grad/ReluGradBgradients_1/discriminator_cat_1/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
¼
4gradients_1/discriminator_cat_1/add_1_grad/Reshape_1Reshape0gradients_1/discriminator_cat_1/add_1_grad/Sum_12gradients_1/discriminator_cat_1/add_1_grad/Shape_1*
T0*
Tshape0
Æ
;gradients_1/discriminator_cat_1/add_1_grad/tuple/group_depsNoOp3^gradients_1/discriminator_cat_1/add_1_grad/Reshape5^gradients_1/discriminator_cat_1/add_1_grad/Reshape_1

Cgradients_1/discriminator_cat_1/add_1_grad/tuple/control_dependencyIdentity2gradients_1/discriminator_cat_1/add_1_grad/Reshape<^gradients_1/discriminator_cat_1/add_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/discriminator_cat_1/add_1_grad/Reshape

Egradients_1/discriminator_cat_1/add_1_grad/tuple/control_dependency_1Identity4gradients_1/discriminator_cat_1/add_1_grad/Reshape_1<^gradients_1/discriminator_cat_1/add_1_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/discriminator_cat_1/add_1_grad/Reshape_1
½
.gradients_1/discriminator/MatMul_1_grad/MatMulMatMul=gradients_1/discriminator/add_1_grad/tuple/control_dependencydiscriminator/w1/read*
transpose_b(*
T0*
transpose_a( 
Å
0gradients_1/discriminator/MatMul_1_grad/MatMul_1MatMuldiscriminator/dropout/mul_1=gradients_1/discriminator/add_1_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
¤
8gradients_1/discriminator/MatMul_1_grad/tuple/group_depsNoOp/^gradients_1/discriminator/MatMul_1_grad/MatMul1^gradients_1/discriminator/MatMul_1_grad/MatMul_1

@gradients_1/discriminator/MatMul_1_grad/tuple/control_dependencyIdentity.gradients_1/discriminator/MatMul_1_grad/MatMul9^gradients_1/discriminator/MatMul_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/discriminator/MatMul_1_grad/MatMul

Bgradients_1/discriminator/MatMul_1_grad/tuple/control_dependency_1Identity0gradients_1/discriminator/MatMul_1_grad/MatMul_19^gradients_1/discriminator/MatMul_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/discriminator/MatMul_1_grad/MatMul_1
Į
0gradients_1/discriminator_1/MatMul_1_grad/MatMulMatMul?gradients_1/discriminator_1/add_1_grad/tuple/control_dependencydiscriminator/w1/read*
transpose_b(*
T0*
transpose_a( 
Ė
2gradients_1/discriminator_1/MatMul_1_grad/MatMul_1MatMuldiscriminator_1/dropout/mul_1?gradients_1/discriminator_1/add_1_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
Ŗ
:gradients_1/discriminator_1/MatMul_1_grad/tuple/group_depsNoOp1^gradients_1/discriminator_1/MatMul_1_grad/MatMul3^gradients_1/discriminator_1/MatMul_1_grad/MatMul_1

Bgradients_1/discriminator_1/MatMul_1_grad/tuple/control_dependencyIdentity0gradients_1/discriminator_1/MatMul_1_grad/MatMul;^gradients_1/discriminator_1/MatMul_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/discriminator_1/MatMul_1_grad/MatMul

Dgradients_1/discriminator_1/MatMul_1_grad/tuple/control_dependency_1Identity2gradients_1/discriminator_1/MatMul_1_grad/MatMul_1;^gradients_1/discriminator_1/MatMul_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/discriminator_1/MatMul_1_grad/MatMul_1
ó
gradients_1/AddN_8AddN?gradients_1/discriminator/add_1_grad/tuple/control_dependency_1Agradients_1/discriminator_1/add_1_grad/tuple/control_dependency_1*
T0*A
_class7
53loc:@gradients_1/discriminator/add_1_grad/Reshape_1*
N
É
2gradients_1/discriminator_cat/MatMul_1_grad/MatMulMatMulAgradients_1/discriminator_cat/add_1_grad/tuple/control_dependencydiscriminator_cat/w1/read*
transpose_b(*
T0*
transpose_a( 
Ń
4gradients_1/discriminator_cat/MatMul_1_grad/MatMul_1MatMuldiscriminator_cat/dropout/mul_1Agradients_1/discriminator_cat/add_1_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
°
<gradients_1/discriminator_cat/MatMul_1_grad/tuple/group_depsNoOp3^gradients_1/discriminator_cat/MatMul_1_grad/MatMul5^gradients_1/discriminator_cat/MatMul_1_grad/MatMul_1

Dgradients_1/discriminator_cat/MatMul_1_grad/tuple/control_dependencyIdentity2gradients_1/discriminator_cat/MatMul_1_grad/MatMul=^gradients_1/discriminator_cat/MatMul_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/discriminator_cat/MatMul_1_grad/MatMul

Fgradients_1/discriminator_cat/MatMul_1_grad/tuple/control_dependency_1Identity4gradients_1/discriminator_cat/MatMul_1_grad/MatMul_1=^gradients_1/discriminator_cat/MatMul_1_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/discriminator_cat/MatMul_1_grad/MatMul_1
Ķ
4gradients_1/discriminator_cat_1/MatMul_1_grad/MatMulMatMulCgradients_1/discriminator_cat_1/add_1_grad/tuple/control_dependencydiscriminator_cat/w1/read*
transpose_b(*
T0*
transpose_a( 
×
6gradients_1/discriminator_cat_1/MatMul_1_grad/MatMul_1MatMul!discriminator_cat_1/dropout/mul_1Cgradients_1/discriminator_cat_1/add_1_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
¶
>gradients_1/discriminator_cat_1/MatMul_1_grad/tuple/group_depsNoOp5^gradients_1/discriminator_cat_1/MatMul_1_grad/MatMul7^gradients_1/discriminator_cat_1/MatMul_1_grad/MatMul_1

Fgradients_1/discriminator_cat_1/MatMul_1_grad/tuple/control_dependencyIdentity4gradients_1/discriminator_cat_1/MatMul_1_grad/MatMul?^gradients_1/discriminator_cat_1/MatMul_1_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/discriminator_cat_1/MatMul_1_grad/MatMul
”
Hgradients_1/discriminator_cat_1/MatMul_1_grad/tuple/control_dependency_1Identity6gradients_1/discriminator_cat_1/MatMul_1_grad/MatMul_1?^gradients_1/discriminator_cat_1/MatMul_1_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients_1/discriminator_cat_1/MatMul_1_grad/MatMul_1
’
gradients_1/AddN_9AddNCgradients_1/discriminator_cat/add_1_grad/tuple/control_dependency_1Egradients_1/discriminator_cat_1/add_1_grad/tuple/control_dependency_1*
T0*E
_class;
97loc:@gradients_1/discriminator_cat/add_1_grad/Reshape_1*
N
o
2gradients_1/discriminator/dropout/mul_1_grad/ShapeShapediscriminator/dropout/mul*
T0*
out_type0
r
4gradients_1/discriminator/dropout/mul_1_grad/Shape_1Shapediscriminator/dropout/Cast*
T0*
out_type0
Ī
Bgradients_1/discriminator/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients_1/discriminator/dropout/mul_1_grad/Shape4gradients_1/discriminator/dropout/mul_1_grad/Shape_1*
T0

0gradients_1/discriminator/dropout/mul_1_grad/MulMul@gradients_1/discriminator/MatMul_1_grad/tuple/control_dependencydiscriminator/dropout/Cast*
T0
Ó
0gradients_1/discriminator/dropout/mul_1_grad/SumSum0gradients_1/discriminator/dropout/mul_1_grad/MulBgradients_1/discriminator/dropout/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
¼
4gradients_1/discriminator/dropout/mul_1_grad/ReshapeReshape0gradients_1/discriminator/dropout/mul_1_grad/Sum2gradients_1/discriminator/dropout/mul_1_grad/Shape*
T0*
Tshape0

2gradients_1/discriminator/dropout/mul_1_grad/Mul_1Muldiscriminator/dropout/mul@gradients_1/discriminator/MatMul_1_grad/tuple/control_dependency*
T0
Ł
2gradients_1/discriminator/dropout/mul_1_grad/Sum_1Sum2gradients_1/discriminator/dropout/mul_1_grad/Mul_1Dgradients_1/discriminator/dropout/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ā
6gradients_1/discriminator/dropout/mul_1_grad/Reshape_1Reshape2gradients_1/discriminator/dropout/mul_1_grad/Sum_14gradients_1/discriminator/dropout/mul_1_grad/Shape_1*
T0*
Tshape0
µ
=gradients_1/discriminator/dropout/mul_1_grad/tuple/group_depsNoOp5^gradients_1/discriminator/dropout/mul_1_grad/Reshape7^gradients_1/discriminator/dropout/mul_1_grad/Reshape_1

Egradients_1/discriminator/dropout/mul_1_grad/tuple/control_dependencyIdentity4gradients_1/discriminator/dropout/mul_1_grad/Reshape>^gradients_1/discriminator/dropout/mul_1_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/discriminator/dropout/mul_1_grad/Reshape

Ggradients_1/discriminator/dropout/mul_1_grad/tuple/control_dependency_1Identity6gradients_1/discriminator/dropout/mul_1_grad/Reshape_1>^gradients_1/discriminator/dropout/mul_1_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients_1/discriminator/dropout/mul_1_grad/Reshape_1
s
4gradients_1/discriminator_1/dropout/mul_1_grad/ShapeShapediscriminator_1/dropout/mul*
T0*
out_type0
v
6gradients_1/discriminator_1/dropout/mul_1_grad/Shape_1Shapediscriminator_1/dropout/Cast*
T0*
out_type0
Ō
Dgradients_1/discriminator_1/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients_1/discriminator_1/dropout/mul_1_grad/Shape6gradients_1/discriminator_1/dropout/mul_1_grad/Shape_1*
T0
¤
2gradients_1/discriminator_1/dropout/mul_1_grad/MulMulBgradients_1/discriminator_1/MatMul_1_grad/tuple/control_dependencydiscriminator_1/dropout/Cast*
T0
Ł
2gradients_1/discriminator_1/dropout/mul_1_grad/SumSum2gradients_1/discriminator_1/dropout/mul_1_grad/MulDgradients_1/discriminator_1/dropout/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
Ā
6gradients_1/discriminator_1/dropout/mul_1_grad/ReshapeReshape2gradients_1/discriminator_1/dropout/mul_1_grad/Sum4gradients_1/discriminator_1/dropout/mul_1_grad/Shape*
T0*
Tshape0
„
4gradients_1/discriminator_1/dropout/mul_1_grad/Mul_1Muldiscriminator_1/dropout/mulBgradients_1/discriminator_1/MatMul_1_grad/tuple/control_dependency*
T0
ß
4gradients_1/discriminator_1/dropout/mul_1_grad/Sum_1Sum4gradients_1/discriminator_1/dropout/mul_1_grad/Mul_1Fgradients_1/discriminator_1/dropout/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Č
8gradients_1/discriminator_1/dropout/mul_1_grad/Reshape_1Reshape4gradients_1/discriminator_1/dropout/mul_1_grad/Sum_16gradients_1/discriminator_1/dropout/mul_1_grad/Shape_1*
T0*
Tshape0
»
?gradients_1/discriminator_1/dropout/mul_1_grad/tuple/group_depsNoOp7^gradients_1/discriminator_1/dropout/mul_1_grad/Reshape9^gradients_1/discriminator_1/dropout/mul_1_grad/Reshape_1
”
Ggradients_1/discriminator_1/dropout/mul_1_grad/tuple/control_dependencyIdentity6gradients_1/discriminator_1/dropout/mul_1_grad/Reshape@^gradients_1/discriminator_1/dropout/mul_1_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients_1/discriminator_1/dropout/mul_1_grad/Reshape
§
Igradients_1/discriminator_1/dropout/mul_1_grad/tuple/control_dependency_1Identity8gradients_1/discriminator_1/dropout/mul_1_grad/Reshape_1@^gradients_1/discriminator_1/dropout/mul_1_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/discriminator_1/dropout/mul_1_grad/Reshape_1
ü
gradients_1/AddN_10AddNBgradients_1/discriminator/MatMul_1_grad/tuple/control_dependency_1Dgradients_1/discriminator_1/MatMul_1_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients_1/discriminator/MatMul_1_grad/MatMul_1*
N
w
6gradients_1/discriminator_cat/dropout/mul_1_grad/ShapeShapediscriminator_cat/dropout/mul*
T0*
out_type0
z
8gradients_1/discriminator_cat/dropout/mul_1_grad/Shape_1Shapediscriminator_cat/dropout/Cast*
T0*
out_type0
Ś
Fgradients_1/discriminator_cat/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_1/discriminator_cat/dropout/mul_1_grad/Shape8gradients_1/discriminator_cat/dropout/mul_1_grad/Shape_1*
T0
Ŗ
4gradients_1/discriminator_cat/dropout/mul_1_grad/MulMulDgradients_1/discriminator_cat/MatMul_1_grad/tuple/control_dependencydiscriminator_cat/dropout/Cast*
T0
ß
4gradients_1/discriminator_cat/dropout/mul_1_grad/SumSum4gradients_1/discriminator_cat/dropout/mul_1_grad/MulFgradients_1/discriminator_cat/dropout/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
Č
8gradients_1/discriminator_cat/dropout/mul_1_grad/ReshapeReshape4gradients_1/discriminator_cat/dropout/mul_1_grad/Sum6gradients_1/discriminator_cat/dropout/mul_1_grad/Shape*
T0*
Tshape0
«
6gradients_1/discriminator_cat/dropout/mul_1_grad/Mul_1Muldiscriminator_cat/dropout/mulDgradients_1/discriminator_cat/MatMul_1_grad/tuple/control_dependency*
T0
å
6gradients_1/discriminator_cat/dropout/mul_1_grad/Sum_1Sum6gradients_1/discriminator_cat/dropout/mul_1_grad/Mul_1Hgradients_1/discriminator_cat/dropout/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ī
:gradients_1/discriminator_cat/dropout/mul_1_grad/Reshape_1Reshape6gradients_1/discriminator_cat/dropout/mul_1_grad/Sum_18gradients_1/discriminator_cat/dropout/mul_1_grad/Shape_1*
T0*
Tshape0
Į
Agradients_1/discriminator_cat/dropout/mul_1_grad/tuple/group_depsNoOp9^gradients_1/discriminator_cat/dropout/mul_1_grad/Reshape;^gradients_1/discriminator_cat/dropout/mul_1_grad/Reshape_1
©
Igradients_1/discriminator_cat/dropout/mul_1_grad/tuple/control_dependencyIdentity8gradients_1/discriminator_cat/dropout/mul_1_grad/ReshapeB^gradients_1/discriminator_cat/dropout/mul_1_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/discriminator_cat/dropout/mul_1_grad/Reshape
Æ
Kgradients_1/discriminator_cat/dropout/mul_1_grad/tuple/control_dependency_1Identity:gradients_1/discriminator_cat/dropout/mul_1_grad/Reshape_1B^gradients_1/discriminator_cat/dropout/mul_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/discriminator_cat/dropout/mul_1_grad/Reshape_1
{
8gradients_1/discriminator_cat_1/dropout/mul_1_grad/ShapeShapediscriminator_cat_1/dropout/mul*
T0*
out_type0
~
:gradients_1/discriminator_cat_1/dropout/mul_1_grad/Shape_1Shape discriminator_cat_1/dropout/Cast*
T0*
out_type0
ą
Hgradients_1/discriminator_cat_1/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients_1/discriminator_cat_1/dropout/mul_1_grad/Shape:gradients_1/discriminator_cat_1/dropout/mul_1_grad/Shape_1*
T0
°
6gradients_1/discriminator_cat_1/dropout/mul_1_grad/MulMulFgradients_1/discriminator_cat_1/MatMul_1_grad/tuple/control_dependency discriminator_cat_1/dropout/Cast*
T0
å
6gradients_1/discriminator_cat_1/dropout/mul_1_grad/SumSum6gradients_1/discriminator_cat_1/dropout/mul_1_grad/MulHgradients_1/discriminator_cat_1/dropout/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
Ī
:gradients_1/discriminator_cat_1/dropout/mul_1_grad/ReshapeReshape6gradients_1/discriminator_cat_1/dropout/mul_1_grad/Sum8gradients_1/discriminator_cat_1/dropout/mul_1_grad/Shape*
T0*
Tshape0
±
8gradients_1/discriminator_cat_1/dropout/mul_1_grad/Mul_1Muldiscriminator_cat_1/dropout/mulFgradients_1/discriminator_cat_1/MatMul_1_grad/tuple/control_dependency*
T0
ė
8gradients_1/discriminator_cat_1/dropout/mul_1_grad/Sum_1Sum8gradients_1/discriminator_cat_1/dropout/mul_1_grad/Mul_1Jgradients_1/discriminator_cat_1/dropout/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ō
<gradients_1/discriminator_cat_1/dropout/mul_1_grad/Reshape_1Reshape8gradients_1/discriminator_cat_1/dropout/mul_1_grad/Sum_1:gradients_1/discriminator_cat_1/dropout/mul_1_grad/Shape_1*
T0*
Tshape0
Ē
Cgradients_1/discriminator_cat_1/dropout/mul_1_grad/tuple/group_depsNoOp;^gradients_1/discriminator_cat_1/dropout/mul_1_grad/Reshape=^gradients_1/discriminator_cat_1/dropout/mul_1_grad/Reshape_1
±
Kgradients_1/discriminator_cat_1/dropout/mul_1_grad/tuple/control_dependencyIdentity:gradients_1/discriminator_cat_1/dropout/mul_1_grad/ReshapeD^gradients_1/discriminator_cat_1/dropout/mul_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/discriminator_cat_1/dropout/mul_1_grad/Reshape
·
Mgradients_1/discriminator_cat_1/dropout/mul_1_grad/tuple/control_dependency_1Identity<gradients_1/discriminator_cat_1/dropout/mul_1_grad/Reshape_1D^gradients_1/discriminator_cat_1/dropout/mul_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_1/discriminator_cat_1/dropout/mul_1_grad/Reshape_1

gradients_1/AddN_11AddNFgradients_1/discriminator_cat/MatMul_1_grad/tuple/control_dependency_1Hgradients_1/discriminator_cat_1/MatMul_1_grad/tuple/control_dependency_1*
T0*G
_class=
;9loc:@gradients_1/discriminator_cat/MatMul_1_grad/MatMul_1*
N
f
0gradients_1/discriminator/dropout/mul_grad/ShapeShapediscriminator/Relu*
T0*
out_type0
s
2gradients_1/discriminator/dropout/mul_grad/Shape_1Shapediscriminator/dropout/truediv*
T0*
out_type0
Č
@gradients_1/discriminator/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs0gradients_1/discriminator/dropout/mul_grad/Shape2gradients_1/discriminator/dropout/mul_grad/Shape_1*
T0
¤
.gradients_1/discriminator/dropout/mul_grad/MulMulEgradients_1/discriminator/dropout/mul_1_grad/tuple/control_dependencydiscriminator/dropout/truediv*
T0
Ķ
.gradients_1/discriminator/dropout/mul_grad/SumSum.gradients_1/discriminator/dropout/mul_grad/Mul@gradients_1/discriminator/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
¶
2gradients_1/discriminator/dropout/mul_grad/ReshapeReshape.gradients_1/discriminator/dropout/mul_grad/Sum0gradients_1/discriminator/dropout/mul_grad/Shape*
T0*
Tshape0

0gradients_1/discriminator/dropout/mul_grad/Mul_1Muldiscriminator/ReluEgradients_1/discriminator/dropout/mul_1_grad/tuple/control_dependency*
T0
Ó
0gradients_1/discriminator/dropout/mul_grad/Sum_1Sum0gradients_1/discriminator/dropout/mul_grad/Mul_1Bgradients_1/discriminator/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
¼
4gradients_1/discriminator/dropout/mul_grad/Reshape_1Reshape0gradients_1/discriminator/dropout/mul_grad/Sum_12gradients_1/discriminator/dropout/mul_grad/Shape_1*
T0*
Tshape0
Æ
;gradients_1/discriminator/dropout/mul_grad/tuple/group_depsNoOp3^gradients_1/discriminator/dropout/mul_grad/Reshape5^gradients_1/discriminator/dropout/mul_grad/Reshape_1

Cgradients_1/discriminator/dropout/mul_grad/tuple/control_dependencyIdentity2gradients_1/discriminator/dropout/mul_grad/Reshape<^gradients_1/discriminator/dropout/mul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/discriminator/dropout/mul_grad/Reshape

Egradients_1/discriminator/dropout/mul_grad/tuple/control_dependency_1Identity4gradients_1/discriminator/dropout/mul_grad/Reshape_1<^gradients_1/discriminator/dropout/mul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/discriminator/dropout/mul_grad/Reshape_1
j
2gradients_1/discriminator_1/dropout/mul_grad/ShapeShapediscriminator_1/Relu*
T0*
out_type0
w
4gradients_1/discriminator_1/dropout/mul_grad/Shape_1Shapediscriminator_1/dropout/truediv*
T0*
out_type0
Ī
Bgradients_1/discriminator_1/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients_1/discriminator_1/dropout/mul_grad/Shape4gradients_1/discriminator_1/dropout/mul_grad/Shape_1*
T0
Ŗ
0gradients_1/discriminator_1/dropout/mul_grad/MulMulGgradients_1/discriminator_1/dropout/mul_1_grad/tuple/control_dependencydiscriminator_1/dropout/truediv*
T0
Ó
0gradients_1/discriminator_1/dropout/mul_grad/SumSum0gradients_1/discriminator_1/dropout/mul_grad/MulBgradients_1/discriminator_1/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
¼
4gradients_1/discriminator_1/dropout/mul_grad/ReshapeReshape0gradients_1/discriminator_1/dropout/mul_grad/Sum2gradients_1/discriminator_1/dropout/mul_grad/Shape*
T0*
Tshape0
”
2gradients_1/discriminator_1/dropout/mul_grad/Mul_1Muldiscriminator_1/ReluGgradients_1/discriminator_1/dropout/mul_1_grad/tuple/control_dependency*
T0
Ł
2gradients_1/discriminator_1/dropout/mul_grad/Sum_1Sum2gradients_1/discriminator_1/dropout/mul_grad/Mul_1Dgradients_1/discriminator_1/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ā
6gradients_1/discriminator_1/dropout/mul_grad/Reshape_1Reshape2gradients_1/discriminator_1/dropout/mul_grad/Sum_14gradients_1/discriminator_1/dropout/mul_grad/Shape_1*
T0*
Tshape0
µ
=gradients_1/discriminator_1/dropout/mul_grad/tuple/group_depsNoOp5^gradients_1/discriminator_1/dropout/mul_grad/Reshape7^gradients_1/discriminator_1/dropout/mul_grad/Reshape_1

Egradients_1/discriminator_1/dropout/mul_grad/tuple/control_dependencyIdentity4gradients_1/discriminator_1/dropout/mul_grad/Reshape>^gradients_1/discriminator_1/dropout/mul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/discriminator_1/dropout/mul_grad/Reshape

Ggradients_1/discriminator_1/dropout/mul_grad/tuple/control_dependency_1Identity6gradients_1/discriminator_1/dropout/mul_grad/Reshape_1>^gradients_1/discriminator_1/dropout/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients_1/discriminator_1/dropout/mul_grad/Reshape_1
n
4gradients_1/discriminator_cat/dropout/mul_grad/ShapeShapediscriminator_cat/Relu*
T0*
out_type0
{
6gradients_1/discriminator_cat/dropout/mul_grad/Shape_1Shape!discriminator_cat/dropout/truediv*
T0*
out_type0
Ō
Dgradients_1/discriminator_cat/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients_1/discriminator_cat/dropout/mul_grad/Shape6gradients_1/discriminator_cat/dropout/mul_grad/Shape_1*
T0
°
2gradients_1/discriminator_cat/dropout/mul_grad/MulMulIgradients_1/discriminator_cat/dropout/mul_1_grad/tuple/control_dependency!discriminator_cat/dropout/truediv*
T0
Ł
2gradients_1/discriminator_cat/dropout/mul_grad/SumSum2gradients_1/discriminator_cat/dropout/mul_grad/MulDgradients_1/discriminator_cat/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
Ā
6gradients_1/discriminator_cat/dropout/mul_grad/ReshapeReshape2gradients_1/discriminator_cat/dropout/mul_grad/Sum4gradients_1/discriminator_cat/dropout/mul_grad/Shape*
T0*
Tshape0
§
4gradients_1/discriminator_cat/dropout/mul_grad/Mul_1Muldiscriminator_cat/ReluIgradients_1/discriminator_cat/dropout/mul_1_grad/tuple/control_dependency*
T0
ß
4gradients_1/discriminator_cat/dropout/mul_grad/Sum_1Sum4gradients_1/discriminator_cat/dropout/mul_grad/Mul_1Fgradients_1/discriminator_cat/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Č
8gradients_1/discriminator_cat/dropout/mul_grad/Reshape_1Reshape4gradients_1/discriminator_cat/dropout/mul_grad/Sum_16gradients_1/discriminator_cat/dropout/mul_grad/Shape_1*
T0*
Tshape0
»
?gradients_1/discriminator_cat/dropout/mul_grad/tuple/group_depsNoOp7^gradients_1/discriminator_cat/dropout/mul_grad/Reshape9^gradients_1/discriminator_cat/dropout/mul_grad/Reshape_1
”
Ggradients_1/discriminator_cat/dropout/mul_grad/tuple/control_dependencyIdentity6gradients_1/discriminator_cat/dropout/mul_grad/Reshape@^gradients_1/discriminator_cat/dropout/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients_1/discriminator_cat/dropout/mul_grad/Reshape
§
Igradients_1/discriminator_cat/dropout/mul_grad/tuple/control_dependency_1Identity8gradients_1/discriminator_cat/dropout/mul_grad/Reshape_1@^gradients_1/discriminator_cat/dropout/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/discriminator_cat/dropout/mul_grad/Reshape_1
r
6gradients_1/discriminator_cat_1/dropout/mul_grad/ShapeShapediscriminator_cat_1/Relu*
T0*
out_type0

8gradients_1/discriminator_cat_1/dropout/mul_grad/Shape_1Shape#discriminator_cat_1/dropout/truediv*
T0*
out_type0
Ś
Fgradients_1/discriminator_cat_1/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_1/discriminator_cat_1/dropout/mul_grad/Shape8gradients_1/discriminator_cat_1/dropout/mul_grad/Shape_1*
T0
¶
4gradients_1/discriminator_cat_1/dropout/mul_grad/MulMulKgradients_1/discriminator_cat_1/dropout/mul_1_grad/tuple/control_dependency#discriminator_cat_1/dropout/truediv*
T0
ß
4gradients_1/discriminator_cat_1/dropout/mul_grad/SumSum4gradients_1/discriminator_cat_1/dropout/mul_grad/MulFgradients_1/discriminator_cat_1/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
Č
8gradients_1/discriminator_cat_1/dropout/mul_grad/ReshapeReshape4gradients_1/discriminator_cat_1/dropout/mul_grad/Sum6gradients_1/discriminator_cat_1/dropout/mul_grad/Shape*
T0*
Tshape0
­
6gradients_1/discriminator_cat_1/dropout/mul_grad/Mul_1Muldiscriminator_cat_1/ReluKgradients_1/discriminator_cat_1/dropout/mul_1_grad/tuple/control_dependency*
T0
å
6gradients_1/discriminator_cat_1/dropout/mul_grad/Sum_1Sum6gradients_1/discriminator_cat_1/dropout/mul_grad/Mul_1Hgradients_1/discriminator_cat_1/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ī
:gradients_1/discriminator_cat_1/dropout/mul_grad/Reshape_1Reshape6gradients_1/discriminator_cat_1/dropout/mul_grad/Sum_18gradients_1/discriminator_cat_1/dropout/mul_grad/Shape_1*
T0*
Tshape0
Į
Agradients_1/discriminator_cat_1/dropout/mul_grad/tuple/group_depsNoOp9^gradients_1/discriminator_cat_1/dropout/mul_grad/Reshape;^gradients_1/discriminator_cat_1/dropout/mul_grad/Reshape_1
©
Igradients_1/discriminator_cat_1/dropout/mul_grad/tuple/control_dependencyIdentity8gradients_1/discriminator_cat_1/dropout/mul_grad/ReshapeB^gradients_1/discriminator_cat_1/dropout/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/discriminator_cat_1/dropout/mul_grad/Reshape
Æ
Kgradients_1/discriminator_cat_1/dropout/mul_grad/tuple/control_dependency_1Identity:gradients_1/discriminator_cat_1/dropout/mul_grad/Reshape_1B^gradients_1/discriminator_cat_1/dropout/mul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/discriminator_cat_1/dropout/mul_grad/Reshape_1

,gradients_1/discriminator/Relu_grad/ReluGradReluGradCgradients_1/discriminator/dropout/mul_grad/tuple/control_dependencydiscriminator/Relu*
T0
 
.gradients_1/discriminator_1/Relu_grad/ReluGradReluGradEgradients_1/discriminator_1/dropout/mul_grad/tuple/control_dependencydiscriminator_1/Relu*
T0
¦
0gradients_1/discriminator_cat/Relu_grad/ReluGradReluGradGgradients_1/discriminator_cat/dropout/mul_grad/tuple/control_dependencydiscriminator_cat/Relu*
T0
¬
2gradients_1/discriminator_cat_1/Relu_grad/ReluGradReluGradIgradients_1/discriminator_cat_1/dropout/mul_grad/tuple/control_dependencydiscriminator_cat_1/Relu*
T0
`
(gradients_1/discriminator/add_grad/ShapeShapediscriminator/MatMul*
T0*
out_type0
c
*gradients_1/discriminator/add_grad/Shape_1Shapediscriminator/b0/read*
T0*
out_type0
°
8gradients_1/discriminator/add_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_1/discriminator/add_grad/Shape*gradients_1/discriminator/add_grad/Shape_1*
T0
»
&gradients_1/discriminator/add_grad/SumSum,gradients_1/discriminator/Relu_grad/ReluGrad8gradients_1/discriminator/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0

*gradients_1/discriminator/add_grad/ReshapeReshape&gradients_1/discriminator/add_grad/Sum(gradients_1/discriminator/add_grad/Shape*
T0*
Tshape0
æ
(gradients_1/discriminator/add_grad/Sum_1Sum,gradients_1/discriminator/Relu_grad/ReluGrad:gradients_1/discriminator/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
¤
,gradients_1/discriminator/add_grad/Reshape_1Reshape(gradients_1/discriminator/add_grad/Sum_1*gradients_1/discriminator/add_grad/Shape_1*
T0*
Tshape0

3gradients_1/discriminator/add_grad/tuple/group_depsNoOp+^gradients_1/discriminator/add_grad/Reshape-^gradients_1/discriminator/add_grad/Reshape_1
ń
;gradients_1/discriminator/add_grad/tuple/control_dependencyIdentity*gradients_1/discriminator/add_grad/Reshape4^gradients_1/discriminator/add_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/discriminator/add_grad/Reshape
÷
=gradients_1/discriminator/add_grad/tuple/control_dependency_1Identity,gradients_1/discriminator/add_grad/Reshape_14^gradients_1/discriminator/add_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/discriminator/add_grad/Reshape_1
d
*gradients_1/discriminator_1/add_grad/ShapeShapediscriminator_1/MatMul*
T0*
out_type0
e
,gradients_1/discriminator_1/add_grad/Shape_1Shapediscriminator/b0/read*
T0*
out_type0
¶
:gradients_1/discriminator_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/discriminator_1/add_grad/Shape,gradients_1/discriminator_1/add_grad/Shape_1*
T0
Į
(gradients_1/discriminator_1/add_grad/SumSum.gradients_1/discriminator_1/Relu_grad/ReluGrad:gradients_1/discriminator_1/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
¤
,gradients_1/discriminator_1/add_grad/ReshapeReshape(gradients_1/discriminator_1/add_grad/Sum*gradients_1/discriminator_1/add_grad/Shape*
T0*
Tshape0
Å
*gradients_1/discriminator_1/add_grad/Sum_1Sum.gradients_1/discriminator_1/Relu_grad/ReluGrad<gradients_1/discriminator_1/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ŗ
.gradients_1/discriminator_1/add_grad/Reshape_1Reshape*gradients_1/discriminator_1/add_grad/Sum_1,gradients_1/discriminator_1/add_grad/Shape_1*
T0*
Tshape0

5gradients_1/discriminator_1/add_grad/tuple/group_depsNoOp-^gradients_1/discriminator_1/add_grad/Reshape/^gradients_1/discriminator_1/add_grad/Reshape_1
ł
=gradients_1/discriminator_1/add_grad/tuple/control_dependencyIdentity,gradients_1/discriminator_1/add_grad/Reshape6^gradients_1/discriminator_1/add_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/discriminator_1/add_grad/Reshape
’
?gradients_1/discriminator_1/add_grad/tuple/control_dependency_1Identity.gradients_1/discriminator_1/add_grad/Reshape_16^gradients_1/discriminator_1/add_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/discriminator_1/add_grad/Reshape_1
h
,gradients_1/discriminator_cat/add_grad/ShapeShapediscriminator_cat/MatMul*
T0*
out_type0
k
.gradients_1/discriminator_cat/add_grad/Shape_1Shapediscriminator_cat/b0/read*
T0*
out_type0
¼
<gradients_1/discriminator_cat/add_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients_1/discriminator_cat/add_grad/Shape.gradients_1/discriminator_cat/add_grad/Shape_1*
T0
Ē
*gradients_1/discriminator_cat/add_grad/SumSum0gradients_1/discriminator_cat/Relu_grad/ReluGrad<gradients_1/discriminator_cat/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
Ŗ
.gradients_1/discriminator_cat/add_grad/ReshapeReshape*gradients_1/discriminator_cat/add_grad/Sum,gradients_1/discriminator_cat/add_grad/Shape*
T0*
Tshape0
Ė
,gradients_1/discriminator_cat/add_grad/Sum_1Sum0gradients_1/discriminator_cat/Relu_grad/ReluGrad>gradients_1/discriminator_cat/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
°
0gradients_1/discriminator_cat/add_grad/Reshape_1Reshape,gradients_1/discriminator_cat/add_grad/Sum_1.gradients_1/discriminator_cat/add_grad/Shape_1*
T0*
Tshape0
£
7gradients_1/discriminator_cat/add_grad/tuple/group_depsNoOp/^gradients_1/discriminator_cat/add_grad/Reshape1^gradients_1/discriminator_cat/add_grad/Reshape_1

?gradients_1/discriminator_cat/add_grad/tuple/control_dependencyIdentity.gradients_1/discriminator_cat/add_grad/Reshape8^gradients_1/discriminator_cat/add_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/discriminator_cat/add_grad/Reshape

Agradients_1/discriminator_cat/add_grad/tuple/control_dependency_1Identity0gradients_1/discriminator_cat/add_grad/Reshape_18^gradients_1/discriminator_cat/add_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/discriminator_cat/add_grad/Reshape_1
l
.gradients_1/discriminator_cat_1/add_grad/ShapeShapediscriminator_cat_1/MatMul*
T0*
out_type0
m
0gradients_1/discriminator_cat_1/add_grad/Shape_1Shapediscriminator_cat/b0/read*
T0*
out_type0
Ā
>gradients_1/discriminator_cat_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients_1/discriminator_cat_1/add_grad/Shape0gradients_1/discriminator_cat_1/add_grad/Shape_1*
T0
Ķ
,gradients_1/discriminator_cat_1/add_grad/SumSum2gradients_1/discriminator_cat_1/Relu_grad/ReluGrad>gradients_1/discriminator_cat_1/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
°
0gradients_1/discriminator_cat_1/add_grad/ReshapeReshape,gradients_1/discriminator_cat_1/add_grad/Sum.gradients_1/discriminator_cat_1/add_grad/Shape*
T0*
Tshape0
Ń
.gradients_1/discriminator_cat_1/add_grad/Sum_1Sum2gradients_1/discriminator_cat_1/Relu_grad/ReluGrad@gradients_1/discriminator_cat_1/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
¶
2gradients_1/discriminator_cat_1/add_grad/Reshape_1Reshape.gradients_1/discriminator_cat_1/add_grad/Sum_10gradients_1/discriminator_cat_1/add_grad/Shape_1*
T0*
Tshape0
©
9gradients_1/discriminator_cat_1/add_grad/tuple/group_depsNoOp1^gradients_1/discriminator_cat_1/add_grad/Reshape3^gradients_1/discriminator_cat_1/add_grad/Reshape_1

Agradients_1/discriminator_cat_1/add_grad/tuple/control_dependencyIdentity0gradients_1/discriminator_cat_1/add_grad/Reshape:^gradients_1/discriminator_cat_1/add_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/discriminator_cat_1/add_grad/Reshape

Cgradients_1/discriminator_cat_1/add_grad/tuple/control_dependency_1Identity2gradients_1/discriminator_cat_1/add_grad/Reshape_1:^gradients_1/discriminator_cat_1/add_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/discriminator_cat_1/add_grad/Reshape_1
¹
,gradients_1/discriminator/MatMul_grad/MatMulMatMul;gradients_1/discriminator/add_grad/tuple/control_dependencydiscriminator/w0/read*
transpose_b(*
T0*
transpose_a( 
²
.gradients_1/discriminator/MatMul_grad/MatMul_1MatMulprior_sample;gradients_1/discriminator/add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(

6gradients_1/discriminator/MatMul_grad/tuple/group_depsNoOp-^gradients_1/discriminator/MatMul_grad/MatMul/^gradients_1/discriminator/MatMul_grad/MatMul_1
ū
>gradients_1/discriminator/MatMul_grad/tuple/control_dependencyIdentity,gradients_1/discriminator/MatMul_grad/MatMul7^gradients_1/discriminator/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/discriminator/MatMul_grad/MatMul

@gradients_1/discriminator/MatMul_grad/tuple/control_dependency_1Identity.gradients_1/discriminator/MatMul_grad/MatMul_17^gradients_1/discriminator/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/discriminator/MatMul_grad/MatMul_1
½
.gradients_1/discriminator_1/MatMul_grad/MatMulMatMul=gradients_1/discriminator_1/add_grad/tuple/control_dependencydiscriminator/w0/read*
transpose_b(*
T0*
transpose_a( 
Ę
0gradients_1/discriminator_1/MatMul_grad/MatMul_1MatMulCNN_encoder_cat/zout/BiasAdd=gradients_1/discriminator_1/add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
¤
8gradients_1/discriminator_1/MatMul_grad/tuple/group_depsNoOp/^gradients_1/discriminator_1/MatMul_grad/MatMul1^gradients_1/discriminator_1/MatMul_grad/MatMul_1

@gradients_1/discriminator_1/MatMul_grad/tuple/control_dependencyIdentity.gradients_1/discriminator_1/MatMul_grad/MatMul9^gradients_1/discriminator_1/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/discriminator_1/MatMul_grad/MatMul

Bgradients_1/discriminator_1/MatMul_grad/tuple/control_dependency_1Identity0gradients_1/discriminator_1/MatMul_grad/MatMul_19^gradients_1/discriminator_1/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/discriminator_1/MatMul_grad/MatMul_1
ī
gradients_1/AddN_12AddN=gradients_1/discriminator/add_grad/tuple/control_dependency_1?gradients_1/discriminator_1/add_grad/tuple/control_dependency_1*
T0*?
_class5
31loc:@gradients_1/discriminator/add_grad/Reshape_1*
N
Å
0gradients_1/discriminator_cat/MatMul_grad/MatMulMatMul?gradients_1/discriminator_cat/add_grad/tuple/control_dependencydiscriminator_cat/w0/read*
transpose_b(*
T0*
transpose_a( 
Ą
2gradients_1/discriminator_cat/MatMul_grad/MatMul_1MatMulprior_sample_label?gradients_1/discriminator_cat/add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
Ŗ
:gradients_1/discriminator_cat/MatMul_grad/tuple/group_depsNoOp1^gradients_1/discriminator_cat/MatMul_grad/MatMul3^gradients_1/discriminator_cat/MatMul_grad/MatMul_1

Bgradients_1/discriminator_cat/MatMul_grad/tuple/control_dependencyIdentity0gradients_1/discriminator_cat/MatMul_grad/MatMul;^gradients_1/discriminator_cat/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/discriminator_cat/MatMul_grad/MatMul

Dgradients_1/discriminator_cat/MatMul_grad/tuple/control_dependency_1Identity2gradients_1/discriminator_cat/MatMul_grad/MatMul_1;^gradients_1/discriminator_cat/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/discriminator_cat/MatMul_grad/MatMul_1
É
2gradients_1/discriminator_cat_1/MatMul_grad/MatMulMatMulAgradients_1/discriminator_cat_1/add_grad/tuple/control_dependencydiscriminator_cat/w0/read*
transpose_b(*
T0*
transpose_a( 
Š
4gradients_1/discriminator_cat_1/MatMul_grad/MatMul_1MatMulCNN_encoder_cat/catout/SoftmaxAgradients_1/discriminator_cat_1/add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
°
<gradients_1/discriminator_cat_1/MatMul_grad/tuple/group_depsNoOp3^gradients_1/discriminator_cat_1/MatMul_grad/MatMul5^gradients_1/discriminator_cat_1/MatMul_grad/MatMul_1

Dgradients_1/discriminator_cat_1/MatMul_grad/tuple/control_dependencyIdentity2gradients_1/discriminator_cat_1/MatMul_grad/MatMul=^gradients_1/discriminator_cat_1/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/discriminator_cat_1/MatMul_grad/MatMul

Fgradients_1/discriminator_cat_1/MatMul_grad/tuple/control_dependency_1Identity4gradients_1/discriminator_cat_1/MatMul_grad/MatMul_1=^gradients_1/discriminator_cat_1/MatMul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/discriminator_cat_1/MatMul_grad/MatMul_1
ś
gradients_1/AddN_13AddNAgradients_1/discriminator_cat/add_grad/tuple/control_dependency_1Cgradients_1/discriminator_cat_1/add_grad/tuple/control_dependency_1*
T0*C
_class9
75loc:@gradients_1/discriminator_cat/add_grad/Reshape_1*
N
Ŗ
9gradients_1/CNN_encoder_cat/zout/BiasAdd_grad/BiasAddGradBiasAddGrad@gradients_1/discriminator_1/MatMul_grad/tuple/control_dependency*
T0*
data_formatNHWC
Å
>gradients_1/CNN_encoder_cat/zout/BiasAdd_grad/tuple/group_depsNoOp:^gradients_1/CNN_encoder_cat/zout/BiasAdd_grad/BiasAddGradA^gradients_1/discriminator_1/MatMul_grad/tuple/control_dependency
”
Fgradients_1/CNN_encoder_cat/zout/BiasAdd_grad/tuple/control_dependencyIdentity@gradients_1/discriminator_1/MatMul_grad/tuple/control_dependency?^gradients_1/CNN_encoder_cat/zout/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/discriminator_1/MatMul_grad/MatMul
§
Hgradients_1/CNN_encoder_cat/zout/BiasAdd_grad/tuple/control_dependency_1Identity9gradients_1/CNN_encoder_cat/zout/BiasAdd_grad/BiasAddGrad?^gradients_1/CNN_encoder_cat/zout/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/CNN_encoder_cat/zout/BiasAdd_grad/BiasAddGrad
ö
gradients_1/AddN_14AddN@gradients_1/discriminator/MatMul_grad/tuple/control_dependency_1Bgradients_1/discriminator_1/MatMul_grad/tuple/control_dependency_1*
T0*A
_class7
53loc:@gradients_1/discriminator/MatMul_grad/MatMul_1*
N
©
3gradients_1/CNN_encoder_cat/catout/Softmax_grad/mulMulDgradients_1/discriminator_cat_1/MatMul_grad/tuple/control_dependencyCNN_encoder_cat/catout/Softmax*
T0
x
Egradients_1/CNN_encoder_cat/catout/Softmax_grad/Sum/reduction_indicesConst*
valueB :
’’’’’’’’’*
dtype0
Ü
3gradients_1/CNN_encoder_cat/catout/Softmax_grad/SumSum3gradients_1/CNN_encoder_cat/catout/Softmax_grad/mulEgradients_1/CNN_encoder_cat/catout/Softmax_grad/Sum/reduction_indices*

Tidx0*
	keep_dims(*
T0
¾
3gradients_1/CNN_encoder_cat/catout/Softmax_grad/subSubDgradients_1/discriminator_cat_1/MatMul_grad/tuple/control_dependency3gradients_1/CNN_encoder_cat/catout/Softmax_grad/Sum*
T0

5gradients_1/CNN_encoder_cat/catout/Softmax_grad/mul_1Mul3gradients_1/CNN_encoder_cat/catout/Softmax_grad/subCNN_encoder_cat/catout/Softmax*
T0

gradients_1/AddN_15AddNDgradients_1/discriminator_cat/MatMul_grad/tuple/control_dependency_1Fgradients_1/discriminator_cat_1/MatMul_grad/tuple/control_dependency_1*
T0*E
_class;
97loc:@gradients_1/discriminator_cat/MatMul_grad/MatMul_1*
N
Ń
3gradients_1/CNN_encoder_cat/zout/MatMul_grad/MatMulMatMulFgradients_1/CNN_encoder_cat/zout/BiasAdd_grad/tuple/control_dependencyCNN_encoder_cat/zout/W/read*
transpose_b(*
T0*
transpose_a( 
Ō
5gradients_1/CNN_encoder_cat/zout/MatMul_grad/MatMul_1MatMulCNN_encoder_cat/zout/ReshapeFgradients_1/CNN_encoder_cat/zout/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
³
=gradients_1/CNN_encoder_cat/zout/MatMul_grad/tuple/group_depsNoOp4^gradients_1/CNN_encoder_cat/zout/MatMul_grad/MatMul6^gradients_1/CNN_encoder_cat/zout/MatMul_grad/MatMul_1

Egradients_1/CNN_encoder_cat/zout/MatMul_grad/tuple/control_dependencyIdentity3gradients_1/CNN_encoder_cat/zout/MatMul_grad/MatMul>^gradients_1/CNN_encoder_cat/zout/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_1/CNN_encoder_cat/zout/MatMul_grad/MatMul

Ggradients_1/CNN_encoder_cat/zout/MatMul_grad/tuple/control_dependency_1Identity5gradients_1/CNN_encoder_cat/zout/MatMul_grad/MatMul_1>^gradients_1/CNN_encoder_cat/zout/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_1/CNN_encoder_cat/zout/MatMul_grad/MatMul_1
”
;gradients_1/CNN_encoder_cat/catout/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients_1/CNN_encoder_cat/catout/Softmax_grad/mul_1*
T0*
data_formatNHWC
¾
@gradients_1/CNN_encoder_cat/catout/BiasAdd_grad/tuple/group_depsNoOp<^gradients_1/CNN_encoder_cat/catout/BiasAdd_grad/BiasAddGrad6^gradients_1/CNN_encoder_cat/catout/Softmax_grad/mul_1
”
Hgradients_1/CNN_encoder_cat/catout/BiasAdd_grad/tuple/control_dependencyIdentity5gradients_1/CNN_encoder_cat/catout/Softmax_grad/mul_1A^gradients_1/CNN_encoder_cat/catout/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_1/CNN_encoder_cat/catout/Softmax_grad/mul_1
Æ
Jgradients_1/CNN_encoder_cat/catout/BiasAdd_grad/tuple/control_dependency_1Identity;gradients_1/CNN_encoder_cat/catout/BiasAdd_grad/BiasAddGradA^gradients_1/CNN_encoder_cat/catout/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/CNN_encoder_cat/catout/BiasAdd_grad/BiasAddGrad
z
3gradients_1/CNN_encoder_cat/zout/Reshape_grad/ShapeShape#CNN_encoder_cat/MaxPool2D_2/MaxPool*
T0*
out_type0
Ó
5gradients_1/CNN_encoder_cat/zout/Reshape_grad/ReshapeReshapeEgradients_1/CNN_encoder_cat/zout/MatMul_grad/tuple/control_dependency3gradients_1/CNN_encoder_cat/zout/Reshape_grad/Shape*
T0*
Tshape0
×
5gradients_1/CNN_encoder_cat/catout/MatMul_grad/MatMulMatMulHgradients_1/CNN_encoder_cat/catout/BiasAdd_grad/tuple/control_dependencyCNN_encoder_cat/catout/W/read*
transpose_b(*
T0*
transpose_a( 
Ś
7gradients_1/CNN_encoder_cat/catout/MatMul_grad/MatMul_1MatMulCNN_encoder_cat/catout/ReshapeHgradients_1/CNN_encoder_cat/catout/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
¹
?gradients_1/CNN_encoder_cat/catout/MatMul_grad/tuple/group_depsNoOp6^gradients_1/CNN_encoder_cat/catout/MatMul_grad/MatMul8^gradients_1/CNN_encoder_cat/catout/MatMul_grad/MatMul_1

Ggradients_1/CNN_encoder_cat/catout/MatMul_grad/tuple/control_dependencyIdentity5gradients_1/CNN_encoder_cat/catout/MatMul_grad/MatMul@^gradients_1/CNN_encoder_cat/catout/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_1/CNN_encoder_cat/catout/MatMul_grad/MatMul
„
Igradients_1/CNN_encoder_cat/catout/MatMul_grad/tuple/control_dependency_1Identity7gradients_1/CNN_encoder_cat/catout/MatMul_grad/MatMul_1@^gradients_1/CNN_encoder_cat/catout/MatMul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_1/CNN_encoder_cat/catout/MatMul_grad/MatMul_1
|
5gradients_1/CNN_encoder_cat/catout/Reshape_grad/ShapeShape#CNN_encoder_cat/MaxPool2D_2/MaxPool*
T0*
out_type0
Ł
7gradients_1/CNN_encoder_cat/catout/Reshape_grad/ReshapeReshapeGgradients_1/CNN_encoder_cat/catout/MatMul_grad/tuple/control_dependency5gradients_1/CNN_encoder_cat/catout/Reshape_grad/Shape*
T0*
Tshape0
ē
gradients_1/AddN_16AddN5gradients_1/CNN_encoder_cat/zout/Reshape_grad/Reshape7gradients_1/CNN_encoder_cat/catout/Reshape_grad/Reshape*
T0*H
_class>
<:loc:@gradients_1/CNN_encoder_cat/zout/Reshape_grad/Reshape*
N

@gradients_1/CNN_encoder_cat/MaxPool2D_2/MaxPool_grad/MaxPoolGradMaxPoolGradCNN_encoder_cat/Conv2D_2/Tanh#CNN_encoder_cat/MaxPool2D_2/MaxPoolgradients_1/AddN_16*
ksize
*
paddingSAME*
T0*
data_formatNHWC*
strides

­
7gradients_1/CNN_encoder_cat/Conv2D_2/Tanh_grad/TanhGradTanhGradCNN_encoder_cat/Conv2D_2/Tanh@gradients_1/CNN_encoder_cat/MaxPool2D_2/MaxPool_grad/MaxPoolGrad*
T0
„
=gradients_1/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients_1/CNN_encoder_cat/Conv2D_2/Tanh_grad/TanhGrad*
T0*
data_formatNHWC
Ä
Bgradients_1/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/group_depsNoOp>^gradients_1/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/BiasAddGrad8^gradients_1/CNN_encoder_cat/Conv2D_2/Tanh_grad/TanhGrad
©
Jgradients_1/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/control_dependencyIdentity7gradients_1/CNN_encoder_cat/Conv2D_2/Tanh_grad/TanhGradC^gradients_1/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_1/CNN_encoder_cat/Conv2D_2/Tanh_grad/TanhGrad
·
Lgradients_1/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/control_dependency_1Identity=gradients_1/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/BiasAddGradC^gradients_1/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/BiasAddGrad
©
7gradients_1/CNN_encoder_cat/Conv2D_2/Conv2D_grad/ShapeNShapeN#CNN_encoder_cat/MaxPool2D_1/MaxPoolCNN_encoder_cat/Conv2D_2/W/read*
T0*
out_type0*
N

Dgradients_1/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput7gradients_1/CNN_encoder_cat/Conv2D_2/Conv2D_grad/ShapeNCNN_encoder_cat/Conv2D_2/W/readJgradients_1/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME

Egradients_1/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter#CNN_encoder_cat/MaxPool2D_1/MaxPool9gradients_1/CNN_encoder_cat/Conv2D_2/Conv2D_grad/ShapeN:1Jgradients_1/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
Ų
Agradients_1/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/group_depsNoOpF^gradients_1/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropFilterE^gradients_1/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropInput
Į
Igradients_1/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/control_dependencyIdentityDgradients_1/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropInputB^gradients_1/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropInput
Å
Kgradients_1/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/control_dependency_1IdentityEgradients_1/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropFilterB^gradients_1/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropFilter
¹
@gradients_1/CNN_encoder_cat/MaxPool2D_1/MaxPool_grad/MaxPoolGradMaxPoolGradCNN_encoder_cat/Conv2D_1/Tanh#CNN_encoder_cat/MaxPool2D_1/MaxPoolIgradients_1/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*
T0*
data_formatNHWC*
strides

­
7gradients_1/CNN_encoder_cat/Conv2D_1/Tanh_grad/TanhGradTanhGradCNN_encoder_cat/Conv2D_1/Tanh@gradients_1/CNN_encoder_cat/MaxPool2D_1/MaxPool_grad/MaxPoolGrad*
T0
„
=gradients_1/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients_1/CNN_encoder_cat/Conv2D_1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC
Ä
Bgradients_1/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/group_depsNoOp>^gradients_1/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/BiasAddGrad8^gradients_1/CNN_encoder_cat/Conv2D_1/Tanh_grad/TanhGrad
©
Jgradients_1/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/control_dependencyIdentity7gradients_1/CNN_encoder_cat/Conv2D_1/Tanh_grad/TanhGradC^gradients_1/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_1/CNN_encoder_cat/Conv2D_1/Tanh_grad/TanhGrad
·
Lgradients_1/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/control_dependency_1Identity=gradients_1/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/BiasAddGradC^gradients_1/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/BiasAddGrad
§
7gradients_1/CNN_encoder_cat/Conv2D_1/Conv2D_grad/ShapeNShapeN!CNN_encoder_cat/MaxPool2D/MaxPoolCNN_encoder_cat/Conv2D_1/W/read*
T0*
out_type0*
N

Dgradients_1/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput7gradients_1/CNN_encoder_cat/Conv2D_1/Conv2D_grad/ShapeNCNN_encoder_cat/Conv2D_1/W/readJgradients_1/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME

Egradients_1/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter!CNN_encoder_cat/MaxPool2D/MaxPool9gradients_1/CNN_encoder_cat/Conv2D_1/Conv2D_grad/ShapeN:1Jgradients_1/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
Ų
Agradients_1/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/group_depsNoOpF^gradients_1/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropFilterE^gradients_1/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropInput
Į
Igradients_1/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/control_dependencyIdentityDgradients_1/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropInputB^gradients_1/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_1/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropInput
Å
Kgradients_1/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/control_dependency_1IdentityEgradients_1/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropFilterB^gradients_1/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_1/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropFilter
³
>gradients_1/CNN_encoder_cat/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradCNN_encoder_cat/Conv2D/Tanh!CNN_encoder_cat/MaxPool2D/MaxPoolIgradients_1/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*
T0*
data_formatNHWC*
strides

§
5gradients_1/CNN_encoder_cat/Conv2D/Tanh_grad/TanhGradTanhGradCNN_encoder_cat/Conv2D/Tanh>gradients_1/CNN_encoder_cat/MaxPool2D/MaxPool_grad/MaxPoolGrad*
T0
”
;gradients_1/CNN_encoder_cat/Conv2D/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients_1/CNN_encoder_cat/Conv2D/Tanh_grad/TanhGrad*
T0*
data_formatNHWC
¾
@gradients_1/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/group_depsNoOp<^gradients_1/CNN_encoder_cat/Conv2D/BiasAdd_grad/BiasAddGrad6^gradients_1/CNN_encoder_cat/Conv2D/Tanh_grad/TanhGrad
”
Hgradients_1/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/control_dependencyIdentity5gradients_1/CNN_encoder_cat/Conv2D/Tanh_grad/TanhGradA^gradients_1/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_1/CNN_encoder_cat/Conv2D/Tanh_grad/TanhGrad
Æ
Jgradients_1/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/control_dependency_1Identity;gradients_1/CNN_encoder_cat/Conv2D/BiasAdd_grad/BiasAddGradA^gradients_1/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/CNN_encoder_cat/Conv2D/BiasAdd_grad/BiasAddGrad

5gradients_1/CNN_encoder_cat/Conv2D/Conv2D_grad/ShapeNShapeNCNN_encoder_cat/ReshapeCNN_encoder_cat/Conv2D/W/read*
T0*
out_type0*
N

Bgradients_1/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput5gradients_1/CNN_encoder_cat/Conv2D/Conv2D_grad/ShapeNCNN_encoder_cat/Conv2D/W/readHgradients_1/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME

Cgradients_1/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterCNN_encoder_cat/Reshape7gradients_1/CNN_encoder_cat/Conv2D/Conv2D_grad/ShapeN:1Hgradients_1/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
Ņ
?gradients_1/CNN_encoder_cat/Conv2D/Conv2D_grad/tuple/group_depsNoOpD^gradients_1/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropFilterC^gradients_1/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropInput
¹
Ggradients_1/CNN_encoder_cat/Conv2D/Conv2D_grad/tuple/control_dependencyIdentityBgradients_1/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropInput@^gradients_1/CNN_encoder_cat/Conv2D/Conv2D_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_1/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropInput
½
Igradients_1/CNN_encoder_cat/Conv2D/Conv2D_grad/tuple/control_dependency_1IdentityCgradients_1/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropFilter@^gradients_1/CNN_encoder_cat/Conv2D/Conv2D_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_1/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropFilter
u
beta1_power_1/initial_valueConst*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
valueB
 *fff?*
dtype0

beta1_power_1
VariableV2*
shape: *
shared_name *+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
dtype0*
	container 
©
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(
c
beta1_power_1/readIdentitybeta1_power_1*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W
u
beta2_power_1/initial_valueConst*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
valueB
 *w¾?*
dtype0

beta2_power_1
VariableV2*
shape: *
shared_name *+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
dtype0*
	container 
©
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(
c
beta2_power_1/readIdentitybeta2_power_1*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W

1CNN_encoder_cat/Conv2D/W/Adam_2/Initializer/zerosConst*%
valueB *    *+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
dtype0
Ø
CNN_encoder_cat/Conv2D/W/Adam_2
VariableV2*
shape: *
shared_name *+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
dtype0*
	container 
ć
&CNN_encoder_cat/Conv2D/W/Adam_2/AssignAssignCNN_encoder_cat/Conv2D/W/Adam_21CNN_encoder_cat/Conv2D/W/Adam_2/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(

$CNN_encoder_cat/Conv2D/W/Adam_2/readIdentityCNN_encoder_cat/Conv2D/W/Adam_2*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W

1CNN_encoder_cat/Conv2D/W/Adam_3/Initializer/zerosConst*%
valueB *    *+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
dtype0
Ø
CNN_encoder_cat/Conv2D/W/Adam_3
VariableV2*
shape: *
shared_name *+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
dtype0*
	container 
ć
&CNN_encoder_cat/Conv2D/W/Adam_3/AssignAssignCNN_encoder_cat/Conv2D/W/Adam_31CNN_encoder_cat/Conv2D/W/Adam_3/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(

$CNN_encoder_cat/Conv2D/W/Adam_3/readIdentityCNN_encoder_cat/Conv2D/W/Adam_3*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W

1CNN_encoder_cat/Conv2D/b/Adam_2/Initializer/zerosConst*
valueB *    *+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
dtype0

CNN_encoder_cat/Conv2D/b/Adam_2
VariableV2*
shape: *
shared_name *+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
dtype0*
	container 
ć
&CNN_encoder_cat/Conv2D/b/Adam_2/AssignAssignCNN_encoder_cat/Conv2D/b/Adam_21CNN_encoder_cat/Conv2D/b/Adam_2/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
validate_shape(

$CNN_encoder_cat/Conv2D/b/Adam_2/readIdentityCNN_encoder_cat/Conv2D/b/Adam_2*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b

1CNN_encoder_cat/Conv2D/b/Adam_3/Initializer/zerosConst*
valueB *    *+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
dtype0

CNN_encoder_cat/Conv2D/b/Adam_3
VariableV2*
shape: *
shared_name *+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
dtype0*
	container 
ć
&CNN_encoder_cat/Conv2D/b/Adam_3/AssignAssignCNN_encoder_cat/Conv2D/b/Adam_31CNN_encoder_cat/Conv2D/b/Adam_3/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
validate_shape(

$CNN_encoder_cat/Conv2D/b/Adam_3/readIdentityCNN_encoder_cat/Conv2D/b/Adam_3*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b
Æ
CCNN_encoder_cat/Conv2D_1/W/Adam_2/Initializer/zeros/shape_as_tensorConst*%
valueB"              *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0

9CNN_encoder_cat/Conv2D_1/W/Adam_2/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0

3CNN_encoder_cat/Conv2D_1/W/Adam_2/Initializer/zerosFillCCNN_encoder_cat/Conv2D_1/W/Adam_2/Initializer/zeros/shape_as_tensor9CNN_encoder_cat/Conv2D_1/W/Adam_2/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W
¬
!CNN_encoder_cat/Conv2D_1/W/Adam_2
VariableV2*
shape:  *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_1/W/Adam_2/AssignAssign!CNN_encoder_cat/Conv2D_1/W/Adam_23CNN_encoder_cat/Conv2D_1/W/Adam_2/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
validate_shape(

&CNN_encoder_cat/Conv2D_1/W/Adam_2/readIdentity!CNN_encoder_cat/Conv2D_1/W/Adam_2*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W
Æ
CCNN_encoder_cat/Conv2D_1/W/Adam_3/Initializer/zeros/shape_as_tensorConst*%
valueB"              *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0

9CNN_encoder_cat/Conv2D_1/W/Adam_3/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0

3CNN_encoder_cat/Conv2D_1/W/Adam_3/Initializer/zerosFillCCNN_encoder_cat/Conv2D_1/W/Adam_3/Initializer/zeros/shape_as_tensor9CNN_encoder_cat/Conv2D_1/W/Adam_3/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W
¬
!CNN_encoder_cat/Conv2D_1/W/Adam_3
VariableV2*
shape:  *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_1/W/Adam_3/AssignAssign!CNN_encoder_cat/Conv2D_1/W/Adam_33CNN_encoder_cat/Conv2D_1/W/Adam_3/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
validate_shape(

&CNN_encoder_cat/Conv2D_1/W/Adam_3/readIdentity!CNN_encoder_cat/Conv2D_1/W/Adam_3*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W

3CNN_encoder_cat/Conv2D_1/b/Adam_2/Initializer/zerosConst*
valueB *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
dtype0
 
!CNN_encoder_cat/Conv2D_1/b/Adam_2
VariableV2*
shape: *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_1/b/Adam_2/AssignAssign!CNN_encoder_cat/Conv2D_1/b/Adam_23CNN_encoder_cat/Conv2D_1/b/Adam_2/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
validate_shape(

&CNN_encoder_cat/Conv2D_1/b/Adam_2/readIdentity!CNN_encoder_cat/Conv2D_1/b/Adam_2*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b

3CNN_encoder_cat/Conv2D_1/b/Adam_3/Initializer/zerosConst*
valueB *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
dtype0
 
!CNN_encoder_cat/Conv2D_1/b/Adam_3
VariableV2*
shape: *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_1/b/Adam_3/AssignAssign!CNN_encoder_cat/Conv2D_1/b/Adam_33CNN_encoder_cat/Conv2D_1/b/Adam_3/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
validate_shape(

&CNN_encoder_cat/Conv2D_1/b/Adam_3/readIdentity!CNN_encoder_cat/Conv2D_1/b/Adam_3*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b
Æ
CCNN_encoder_cat/Conv2D_2/W/Adam_2/Initializer/zeros/shape_as_tensorConst*%
valueB"              *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0

9CNN_encoder_cat/Conv2D_2/W/Adam_2/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0

3CNN_encoder_cat/Conv2D_2/W/Adam_2/Initializer/zerosFillCCNN_encoder_cat/Conv2D_2/W/Adam_2/Initializer/zeros/shape_as_tensor9CNN_encoder_cat/Conv2D_2/W/Adam_2/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W
¬
!CNN_encoder_cat/Conv2D_2/W/Adam_2
VariableV2*
shape:  *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_2/W/Adam_2/AssignAssign!CNN_encoder_cat/Conv2D_2/W/Adam_23CNN_encoder_cat/Conv2D_2/W/Adam_2/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
validate_shape(

&CNN_encoder_cat/Conv2D_2/W/Adam_2/readIdentity!CNN_encoder_cat/Conv2D_2/W/Adam_2*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W
Æ
CCNN_encoder_cat/Conv2D_2/W/Adam_3/Initializer/zeros/shape_as_tensorConst*%
valueB"              *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0

9CNN_encoder_cat/Conv2D_2/W/Adam_3/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0

3CNN_encoder_cat/Conv2D_2/W/Adam_3/Initializer/zerosFillCCNN_encoder_cat/Conv2D_2/W/Adam_3/Initializer/zeros/shape_as_tensor9CNN_encoder_cat/Conv2D_2/W/Adam_3/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W
¬
!CNN_encoder_cat/Conv2D_2/W/Adam_3
VariableV2*
shape:  *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_2/W/Adam_3/AssignAssign!CNN_encoder_cat/Conv2D_2/W/Adam_33CNN_encoder_cat/Conv2D_2/W/Adam_3/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
validate_shape(

&CNN_encoder_cat/Conv2D_2/W/Adam_3/readIdentity!CNN_encoder_cat/Conv2D_2/W/Adam_3*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W

3CNN_encoder_cat/Conv2D_2/b/Adam_2/Initializer/zerosConst*
valueB *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
dtype0
 
!CNN_encoder_cat/Conv2D_2/b/Adam_2
VariableV2*
shape: *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_2/b/Adam_2/AssignAssign!CNN_encoder_cat/Conv2D_2/b/Adam_23CNN_encoder_cat/Conv2D_2/b/Adam_2/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
validate_shape(

&CNN_encoder_cat/Conv2D_2/b/Adam_2/readIdentity!CNN_encoder_cat/Conv2D_2/b/Adam_2*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b

3CNN_encoder_cat/Conv2D_2/b/Adam_3/Initializer/zerosConst*
valueB *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
dtype0
 
!CNN_encoder_cat/Conv2D_2/b/Adam_3
VariableV2*
shape: *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_2/b/Adam_3/AssignAssign!CNN_encoder_cat/Conv2D_2/b/Adam_33CNN_encoder_cat/Conv2D_2/b/Adam_3/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
validate_shape(

&CNN_encoder_cat/Conv2D_2/b/Adam_3/readIdentity!CNN_encoder_cat/Conv2D_2/b/Adam_3*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b
£
ACNN_encoder_cat/catout/W/Adam_2/Initializer/zeros/shape_as_tensorConst*
valueB"     *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0

7CNN_encoder_cat/catout/W/Adam_2/Initializer/zeros/ConstConst*
valueB
 *    *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0
ż
1CNN_encoder_cat/catout/W/Adam_2/Initializer/zerosFillACNN_encoder_cat/catout/W/Adam_2/Initializer/zeros/shape_as_tensor7CNN_encoder_cat/catout/W/Adam_2/Initializer/zeros/Const*
T0*

index_type0*+
_class!
loc:@CNN_encoder_cat/catout/W
”
CNN_encoder_cat/catout/W/Adam_2
VariableV2*
shape:	*
shared_name *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0*
	container 
ć
&CNN_encoder_cat/catout/W/Adam_2/AssignAssignCNN_encoder_cat/catout/W/Adam_21CNN_encoder_cat/catout/W/Adam_2/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W*
validate_shape(

$CNN_encoder_cat/catout/W/Adam_2/readIdentityCNN_encoder_cat/catout/W/Adam_2*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W
£
ACNN_encoder_cat/catout/W/Adam_3/Initializer/zeros/shape_as_tensorConst*
valueB"     *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0

7CNN_encoder_cat/catout/W/Adam_3/Initializer/zeros/ConstConst*
valueB
 *    *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0
ż
1CNN_encoder_cat/catout/W/Adam_3/Initializer/zerosFillACNN_encoder_cat/catout/W/Adam_3/Initializer/zeros/shape_as_tensor7CNN_encoder_cat/catout/W/Adam_3/Initializer/zeros/Const*
T0*

index_type0*+
_class!
loc:@CNN_encoder_cat/catout/W
”
CNN_encoder_cat/catout/W/Adam_3
VariableV2*
shape:	*
shared_name *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0*
	container 
ć
&CNN_encoder_cat/catout/W/Adam_3/AssignAssignCNN_encoder_cat/catout/W/Adam_31CNN_encoder_cat/catout/W/Adam_3/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W*
validate_shape(

$CNN_encoder_cat/catout/W/Adam_3/readIdentityCNN_encoder_cat/catout/W/Adam_3*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W

1CNN_encoder_cat/catout/b/Adam_2/Initializer/zerosConst*
valueB*    *+
_class!
loc:@CNN_encoder_cat/catout/b*
dtype0

CNN_encoder_cat/catout/b/Adam_2
VariableV2*
shape:*
shared_name *+
_class!
loc:@CNN_encoder_cat/catout/b*
dtype0*
	container 
ć
&CNN_encoder_cat/catout/b/Adam_2/AssignAssignCNN_encoder_cat/catout/b/Adam_21CNN_encoder_cat/catout/b/Adam_2/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b*
validate_shape(

$CNN_encoder_cat/catout/b/Adam_2/readIdentityCNN_encoder_cat/catout/b/Adam_2*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b

1CNN_encoder_cat/catout/b/Adam_3/Initializer/zerosConst*
valueB*    *+
_class!
loc:@CNN_encoder_cat/catout/b*
dtype0

CNN_encoder_cat/catout/b/Adam_3
VariableV2*
shape:*
shared_name *+
_class!
loc:@CNN_encoder_cat/catout/b*
dtype0*
	container 
ć
&CNN_encoder_cat/catout/b/Adam_3/AssignAssignCNN_encoder_cat/catout/b/Adam_31CNN_encoder_cat/catout/b/Adam_3/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b*
validate_shape(

$CNN_encoder_cat/catout/b/Adam_3/readIdentityCNN_encoder_cat/catout/b/Adam_3*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b

?CNN_encoder_cat/zout/W/Adam_2/Initializer/zeros/shape_as_tensorConst*
valueB"  2   *)
_class
loc:@CNN_encoder_cat/zout/W*
dtype0

5CNN_encoder_cat/zout/W/Adam_2/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@CNN_encoder_cat/zout/W*
dtype0
õ
/CNN_encoder_cat/zout/W/Adam_2/Initializer/zerosFill?CNN_encoder_cat/zout/W/Adam_2/Initializer/zeros/shape_as_tensor5CNN_encoder_cat/zout/W/Adam_2/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@CNN_encoder_cat/zout/W

CNN_encoder_cat/zout/W/Adam_2
VariableV2*
shape:	2*
shared_name *)
_class
loc:@CNN_encoder_cat/zout/W*
dtype0*
	container 
Ū
$CNN_encoder_cat/zout/W/Adam_2/AssignAssignCNN_encoder_cat/zout/W/Adam_2/CNN_encoder_cat/zout/W/Adam_2/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/W*
validate_shape(

"CNN_encoder_cat/zout/W/Adam_2/readIdentityCNN_encoder_cat/zout/W/Adam_2*
T0*)
_class
loc:@CNN_encoder_cat/zout/W

?CNN_encoder_cat/zout/W/Adam_3/Initializer/zeros/shape_as_tensorConst*
valueB"  2   *)
_class
loc:@CNN_encoder_cat/zout/W*
dtype0

5CNN_encoder_cat/zout/W/Adam_3/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@CNN_encoder_cat/zout/W*
dtype0
õ
/CNN_encoder_cat/zout/W/Adam_3/Initializer/zerosFill?CNN_encoder_cat/zout/W/Adam_3/Initializer/zeros/shape_as_tensor5CNN_encoder_cat/zout/W/Adam_3/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@CNN_encoder_cat/zout/W

CNN_encoder_cat/zout/W/Adam_3
VariableV2*
shape:	2*
shared_name *)
_class
loc:@CNN_encoder_cat/zout/W*
dtype0*
	container 
Ū
$CNN_encoder_cat/zout/W/Adam_3/AssignAssignCNN_encoder_cat/zout/W/Adam_3/CNN_encoder_cat/zout/W/Adam_3/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/W*
validate_shape(

"CNN_encoder_cat/zout/W/Adam_3/readIdentityCNN_encoder_cat/zout/W/Adam_3*
T0*)
_class
loc:@CNN_encoder_cat/zout/W

/CNN_encoder_cat/zout/b/Adam_2/Initializer/zerosConst*
valueB2*    *)
_class
loc:@CNN_encoder_cat/zout/b*
dtype0

CNN_encoder_cat/zout/b/Adam_2
VariableV2*
shape:2*
shared_name *)
_class
loc:@CNN_encoder_cat/zout/b*
dtype0*
	container 
Ū
$CNN_encoder_cat/zout/b/Adam_2/AssignAssignCNN_encoder_cat/zout/b/Adam_2/CNN_encoder_cat/zout/b/Adam_2/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/b*
validate_shape(

"CNN_encoder_cat/zout/b/Adam_2/readIdentityCNN_encoder_cat/zout/b/Adam_2*
T0*)
_class
loc:@CNN_encoder_cat/zout/b

/CNN_encoder_cat/zout/b/Adam_3/Initializer/zerosConst*
valueB2*    *)
_class
loc:@CNN_encoder_cat/zout/b*
dtype0

CNN_encoder_cat/zout/b/Adam_3
VariableV2*
shape:2*
shared_name *)
_class
loc:@CNN_encoder_cat/zout/b*
dtype0*
	container 
Ū
$CNN_encoder_cat/zout/b/Adam_3/AssignAssignCNN_encoder_cat/zout/b/Adam_3/CNN_encoder_cat/zout/b/Adam_3/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/b*
validate_shape(

"CNN_encoder_cat/zout/b/Adam_3/readIdentityCNN_encoder_cat/zout/b/Adam_3*
T0*)
_class
loc:@CNN_encoder_cat/zout/b

7discriminator/w0/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"2      *#
_class
loc:@discriminator/w0*
dtype0

-discriminator/w0/Adam/Initializer/zeros/ConstConst*
valueB
 *    *#
_class
loc:@discriminator/w0*
dtype0
×
'discriminator/w0/Adam/Initializer/zerosFill7discriminator/w0/Adam/Initializer/zeros/shape_as_tensor-discriminator/w0/Adam/Initializer/zeros/Const*
T0*

index_type0*#
_class
loc:@discriminator/w0

discriminator/w0/Adam
VariableV2*
shape:	2*
shared_name *#
_class
loc:@discriminator/w0*
dtype0*
	container 
½
discriminator/w0/Adam/AssignAssigndiscriminator/w0/Adam'discriminator/w0/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@discriminator/w0*
validate_shape(
k
discriminator/w0/Adam/readIdentitydiscriminator/w0/Adam*
T0*#
_class
loc:@discriminator/w0

9discriminator/w0/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"2      *#
_class
loc:@discriminator/w0*
dtype0

/discriminator/w0/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *#
_class
loc:@discriminator/w0*
dtype0
Ż
)discriminator/w0/Adam_1/Initializer/zerosFill9discriminator/w0/Adam_1/Initializer/zeros/shape_as_tensor/discriminator/w0/Adam_1/Initializer/zeros/Const*
T0*

index_type0*#
_class
loc:@discriminator/w0

discriminator/w0/Adam_1
VariableV2*
shape:	2*
shared_name *#
_class
loc:@discriminator/w0*
dtype0*
	container 
Ć
discriminator/w0/Adam_1/AssignAssigndiscriminator/w0/Adam_1)discriminator/w0/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@discriminator/w0*
validate_shape(
o
discriminator/w0/Adam_1/readIdentitydiscriminator/w0/Adam_1*
T0*#
_class
loc:@discriminator/w0
~
'discriminator/b0/Adam/Initializer/zerosConst*
valueB*    *#
_class
loc:@discriminator/b0*
dtype0

discriminator/b0/Adam
VariableV2*
shape:*
shared_name *#
_class
loc:@discriminator/b0*
dtype0*
	container 
½
discriminator/b0/Adam/AssignAssigndiscriminator/b0/Adam'discriminator/b0/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@discriminator/b0*
validate_shape(
k
discriminator/b0/Adam/readIdentitydiscriminator/b0/Adam*
T0*#
_class
loc:@discriminator/b0

)discriminator/b0/Adam_1/Initializer/zerosConst*
valueB*    *#
_class
loc:@discriminator/b0*
dtype0

discriminator/b0/Adam_1
VariableV2*
shape:*
shared_name *#
_class
loc:@discriminator/b0*
dtype0*
	container 
Ć
discriminator/b0/Adam_1/AssignAssigndiscriminator/b0/Adam_1)discriminator/b0/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@discriminator/b0*
validate_shape(
o
discriminator/b0/Adam_1/readIdentitydiscriminator/b0/Adam_1*
T0*#
_class
loc:@discriminator/b0

7discriminator/w1/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *#
_class
loc:@discriminator/w1*
dtype0

-discriminator/w1/Adam/Initializer/zeros/ConstConst*
valueB
 *    *#
_class
loc:@discriminator/w1*
dtype0
×
'discriminator/w1/Adam/Initializer/zerosFill7discriminator/w1/Adam/Initializer/zeros/shape_as_tensor-discriminator/w1/Adam/Initializer/zeros/Const*
T0*

index_type0*#
_class
loc:@discriminator/w1

discriminator/w1/Adam
VariableV2*
shape:
*
shared_name *#
_class
loc:@discriminator/w1*
dtype0*
	container 
½
discriminator/w1/Adam/AssignAssigndiscriminator/w1/Adam'discriminator/w1/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@discriminator/w1*
validate_shape(
k
discriminator/w1/Adam/readIdentitydiscriminator/w1/Adam*
T0*#
_class
loc:@discriminator/w1

9discriminator/w1/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *#
_class
loc:@discriminator/w1*
dtype0

/discriminator/w1/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *#
_class
loc:@discriminator/w1*
dtype0
Ż
)discriminator/w1/Adam_1/Initializer/zerosFill9discriminator/w1/Adam_1/Initializer/zeros/shape_as_tensor/discriminator/w1/Adam_1/Initializer/zeros/Const*
T0*

index_type0*#
_class
loc:@discriminator/w1

discriminator/w1/Adam_1
VariableV2*
shape:
*
shared_name *#
_class
loc:@discriminator/w1*
dtype0*
	container 
Ć
discriminator/w1/Adam_1/AssignAssigndiscriminator/w1/Adam_1)discriminator/w1/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@discriminator/w1*
validate_shape(
o
discriminator/w1/Adam_1/readIdentitydiscriminator/w1/Adam_1*
T0*#
_class
loc:@discriminator/w1
~
'discriminator/b1/Adam/Initializer/zerosConst*
valueB*    *#
_class
loc:@discriminator/b1*
dtype0

discriminator/b1/Adam
VariableV2*
shape:*
shared_name *#
_class
loc:@discriminator/b1*
dtype0*
	container 
½
discriminator/b1/Adam/AssignAssigndiscriminator/b1/Adam'discriminator/b1/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@discriminator/b1*
validate_shape(
k
discriminator/b1/Adam/readIdentitydiscriminator/b1/Adam*
T0*#
_class
loc:@discriminator/b1

)discriminator/b1/Adam_1/Initializer/zerosConst*
valueB*    *#
_class
loc:@discriminator/b1*
dtype0

discriminator/b1/Adam_1
VariableV2*
shape:*
shared_name *#
_class
loc:@discriminator/b1*
dtype0*
	container 
Ć
discriminator/b1/Adam_1/AssignAssigndiscriminator/b1/Adam_1)discriminator/b1/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@discriminator/b1*
validate_shape(
o
discriminator/b1/Adam_1/readIdentitydiscriminator/b1/Adam_1*
T0*#
_class
loc:@discriminator/b1

'discriminator/wo/Adam/Initializer/zerosConst*
valueB	*    *#
_class
loc:@discriminator/wo*
dtype0

discriminator/wo/Adam
VariableV2*
shape:	*
shared_name *#
_class
loc:@discriminator/wo*
dtype0*
	container 
½
discriminator/wo/Adam/AssignAssigndiscriminator/wo/Adam'discriminator/wo/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@discriminator/wo*
validate_shape(
k
discriminator/wo/Adam/readIdentitydiscriminator/wo/Adam*
T0*#
_class
loc:@discriminator/wo

)discriminator/wo/Adam_1/Initializer/zerosConst*
valueB	*    *#
_class
loc:@discriminator/wo*
dtype0

discriminator/wo/Adam_1
VariableV2*
shape:	*
shared_name *#
_class
loc:@discriminator/wo*
dtype0*
	container 
Ć
discriminator/wo/Adam_1/AssignAssigndiscriminator/wo/Adam_1)discriminator/wo/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@discriminator/wo*
validate_shape(
o
discriminator/wo/Adam_1/readIdentitydiscriminator/wo/Adam_1*
T0*#
_class
loc:@discriminator/wo
}
'discriminator/bo/Adam/Initializer/zerosConst*
valueB*    *#
_class
loc:@discriminator/bo*
dtype0

discriminator/bo/Adam
VariableV2*
shape:*
shared_name *#
_class
loc:@discriminator/bo*
dtype0*
	container 
½
discriminator/bo/Adam/AssignAssigndiscriminator/bo/Adam'discriminator/bo/Adam/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@discriminator/bo*
validate_shape(
k
discriminator/bo/Adam/readIdentitydiscriminator/bo/Adam*
T0*#
_class
loc:@discriminator/bo

)discriminator/bo/Adam_1/Initializer/zerosConst*
valueB*    *#
_class
loc:@discriminator/bo*
dtype0

discriminator/bo/Adam_1
VariableV2*
shape:*
shared_name *#
_class
loc:@discriminator/bo*
dtype0*
	container 
Ć
discriminator/bo/Adam_1/AssignAssigndiscriminator/bo/Adam_1)discriminator/bo/Adam_1/Initializer/zeros*
use_locking(*
T0*#
_class
loc:@discriminator/bo*
validate_shape(
o
discriminator/bo/Adam_1/readIdentitydiscriminator/bo/Adam_1*
T0*#
_class
loc:@discriminator/bo

+discriminator_cat/w0/Adam/Initializer/zerosConst*
valueB	*    *'
_class
loc:@discriminator_cat/w0*
dtype0

discriminator_cat/w0/Adam
VariableV2*
shape:	*
shared_name *'
_class
loc:@discriminator_cat/w0*
dtype0*
	container 
Ķ
 discriminator_cat/w0/Adam/AssignAssigndiscriminator_cat/w0/Adam+discriminator_cat/w0/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@discriminator_cat/w0*
validate_shape(
w
discriminator_cat/w0/Adam/readIdentitydiscriminator_cat/w0/Adam*
T0*'
_class
loc:@discriminator_cat/w0

-discriminator_cat/w0/Adam_1/Initializer/zerosConst*
valueB	*    *'
_class
loc:@discriminator_cat/w0*
dtype0

discriminator_cat/w0/Adam_1
VariableV2*
shape:	*
shared_name *'
_class
loc:@discriminator_cat/w0*
dtype0*
	container 
Ó
"discriminator_cat/w0/Adam_1/AssignAssigndiscriminator_cat/w0/Adam_1-discriminator_cat/w0/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@discriminator_cat/w0*
validate_shape(
{
 discriminator_cat/w0/Adam_1/readIdentitydiscriminator_cat/w0/Adam_1*
T0*'
_class
loc:@discriminator_cat/w0

+discriminator_cat/b0/Adam/Initializer/zerosConst*
valueB*    *'
_class
loc:@discriminator_cat/b0*
dtype0

discriminator_cat/b0/Adam
VariableV2*
shape:*
shared_name *'
_class
loc:@discriminator_cat/b0*
dtype0*
	container 
Ķ
 discriminator_cat/b0/Adam/AssignAssigndiscriminator_cat/b0/Adam+discriminator_cat/b0/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@discriminator_cat/b0*
validate_shape(
w
discriminator_cat/b0/Adam/readIdentitydiscriminator_cat/b0/Adam*
T0*'
_class
loc:@discriminator_cat/b0

-discriminator_cat/b0/Adam_1/Initializer/zerosConst*
valueB*    *'
_class
loc:@discriminator_cat/b0*
dtype0

discriminator_cat/b0/Adam_1
VariableV2*
shape:*
shared_name *'
_class
loc:@discriminator_cat/b0*
dtype0*
	container 
Ó
"discriminator_cat/b0/Adam_1/AssignAssigndiscriminator_cat/b0/Adam_1-discriminator_cat/b0/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@discriminator_cat/b0*
validate_shape(
{
 discriminator_cat/b0/Adam_1/readIdentitydiscriminator_cat/b0/Adam_1*
T0*'
_class
loc:@discriminator_cat/b0

;discriminator_cat/w1/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *'
_class
loc:@discriminator_cat/w1*
dtype0

1discriminator_cat/w1/Adam/Initializer/zeros/ConstConst*
valueB
 *    *'
_class
loc:@discriminator_cat/w1*
dtype0
ē
+discriminator_cat/w1/Adam/Initializer/zerosFill;discriminator_cat/w1/Adam/Initializer/zeros/shape_as_tensor1discriminator_cat/w1/Adam/Initializer/zeros/Const*
T0*

index_type0*'
_class
loc:@discriminator_cat/w1

discriminator_cat/w1/Adam
VariableV2*
shape:
*
shared_name *'
_class
loc:@discriminator_cat/w1*
dtype0*
	container 
Ķ
 discriminator_cat/w1/Adam/AssignAssigndiscriminator_cat/w1/Adam+discriminator_cat/w1/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@discriminator_cat/w1*
validate_shape(
w
discriminator_cat/w1/Adam/readIdentitydiscriminator_cat/w1/Adam*
T0*'
_class
loc:@discriminator_cat/w1

=discriminator_cat/w1/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *'
_class
loc:@discriminator_cat/w1*
dtype0

3discriminator_cat/w1/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *'
_class
loc:@discriminator_cat/w1*
dtype0
ķ
-discriminator_cat/w1/Adam_1/Initializer/zerosFill=discriminator_cat/w1/Adam_1/Initializer/zeros/shape_as_tensor3discriminator_cat/w1/Adam_1/Initializer/zeros/Const*
T0*

index_type0*'
_class
loc:@discriminator_cat/w1

discriminator_cat/w1/Adam_1
VariableV2*
shape:
*
shared_name *'
_class
loc:@discriminator_cat/w1*
dtype0*
	container 
Ó
"discriminator_cat/w1/Adam_1/AssignAssigndiscriminator_cat/w1/Adam_1-discriminator_cat/w1/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@discriminator_cat/w1*
validate_shape(
{
 discriminator_cat/w1/Adam_1/readIdentitydiscriminator_cat/w1/Adam_1*
T0*'
_class
loc:@discriminator_cat/w1

+discriminator_cat/b1/Adam/Initializer/zerosConst*
valueB*    *'
_class
loc:@discriminator_cat/b1*
dtype0

discriminator_cat/b1/Adam
VariableV2*
shape:*
shared_name *'
_class
loc:@discriminator_cat/b1*
dtype0*
	container 
Ķ
 discriminator_cat/b1/Adam/AssignAssigndiscriminator_cat/b1/Adam+discriminator_cat/b1/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@discriminator_cat/b1*
validate_shape(
w
discriminator_cat/b1/Adam/readIdentitydiscriminator_cat/b1/Adam*
T0*'
_class
loc:@discriminator_cat/b1

-discriminator_cat/b1/Adam_1/Initializer/zerosConst*
valueB*    *'
_class
loc:@discriminator_cat/b1*
dtype0

discriminator_cat/b1/Adam_1
VariableV2*
shape:*
shared_name *'
_class
loc:@discriminator_cat/b1*
dtype0*
	container 
Ó
"discriminator_cat/b1/Adam_1/AssignAssigndiscriminator_cat/b1/Adam_1-discriminator_cat/b1/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@discriminator_cat/b1*
validate_shape(
{
 discriminator_cat/b1/Adam_1/readIdentitydiscriminator_cat/b1/Adam_1*
T0*'
_class
loc:@discriminator_cat/b1

+discriminator_cat/wo/Adam/Initializer/zerosConst*
valueB	*    *'
_class
loc:@discriminator_cat/wo*
dtype0

discriminator_cat/wo/Adam
VariableV2*
shape:	*
shared_name *'
_class
loc:@discriminator_cat/wo*
dtype0*
	container 
Ķ
 discriminator_cat/wo/Adam/AssignAssigndiscriminator_cat/wo/Adam+discriminator_cat/wo/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@discriminator_cat/wo*
validate_shape(
w
discriminator_cat/wo/Adam/readIdentitydiscriminator_cat/wo/Adam*
T0*'
_class
loc:@discriminator_cat/wo

-discriminator_cat/wo/Adam_1/Initializer/zerosConst*
valueB	*    *'
_class
loc:@discriminator_cat/wo*
dtype0

discriminator_cat/wo/Adam_1
VariableV2*
shape:	*
shared_name *'
_class
loc:@discriminator_cat/wo*
dtype0*
	container 
Ó
"discriminator_cat/wo/Adam_1/AssignAssigndiscriminator_cat/wo/Adam_1-discriminator_cat/wo/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@discriminator_cat/wo*
validate_shape(
{
 discriminator_cat/wo/Adam_1/readIdentitydiscriminator_cat/wo/Adam_1*
T0*'
_class
loc:@discriminator_cat/wo

+discriminator_cat/bo/Adam/Initializer/zerosConst*
valueB*    *'
_class
loc:@discriminator_cat/bo*
dtype0

discriminator_cat/bo/Adam
VariableV2*
shape:*
shared_name *'
_class
loc:@discriminator_cat/bo*
dtype0*
	container 
Ķ
 discriminator_cat/bo/Adam/AssignAssigndiscriminator_cat/bo/Adam+discriminator_cat/bo/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@discriminator_cat/bo*
validate_shape(
w
discriminator_cat/bo/Adam/readIdentitydiscriminator_cat/bo/Adam*
T0*'
_class
loc:@discriminator_cat/bo

-discriminator_cat/bo/Adam_1/Initializer/zerosConst*
valueB*    *'
_class
loc:@discriminator_cat/bo*
dtype0

discriminator_cat/bo/Adam_1
VariableV2*
shape:*
shared_name *'
_class
loc:@discriminator_cat/bo*
dtype0*
	container 
Ó
"discriminator_cat/bo/Adam_1/AssignAssigndiscriminator_cat/bo/Adam_1-discriminator_cat/bo/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@discriminator_cat/bo*
validate_shape(
{
 discriminator_cat/bo/Adam_1/readIdentitydiscriminator_cat/bo/Adam_1*
T0*'
_class
loc:@discriminator_cat/bo
A
Adam_1/learning_rateConst*
valueB
 *·Ń7*
dtype0
9
Adam_1/beta1Const*
valueB
 *fff?*
dtype0
9
Adam_1/beta2Const*
valueB
 *w¾?*
dtype0
;
Adam_1/epsilonConst*
valueB
 *wĢ+2*
dtype0
«
0Adam_1/update_CNN_encoder_cat/Conv2D/W/ApplyAdam	ApplyAdamCNN_encoder_cat/Conv2D/WCNN_encoder_cat/Conv2D/W/Adam_2CNN_encoder_cat/Conv2D/W/Adam_3beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonIgradients_1/CNN_encoder_cat/Conv2D/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
use_nesterov( 
¬
0Adam_1/update_CNN_encoder_cat/Conv2D/b/ApplyAdam	ApplyAdamCNN_encoder_cat/Conv2D/bCNN_encoder_cat/Conv2D/b/Adam_2CNN_encoder_cat/Conv2D/b/Adam_3beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonJgradients_1/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
use_nesterov( 
·
2Adam_1/update_CNN_encoder_cat/Conv2D_1/W/ApplyAdam	ApplyAdamCNN_encoder_cat/Conv2D_1/W!CNN_encoder_cat/Conv2D_1/W/Adam_2!CNN_encoder_cat/Conv2D_1/W/Adam_3beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonKgradients_1/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
use_nesterov( 
ø
2Adam_1/update_CNN_encoder_cat/Conv2D_1/b/ApplyAdam	ApplyAdamCNN_encoder_cat/Conv2D_1/b!CNN_encoder_cat/Conv2D_1/b/Adam_2!CNN_encoder_cat/Conv2D_1/b/Adam_3beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonLgradients_1/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
use_nesterov( 
·
2Adam_1/update_CNN_encoder_cat/Conv2D_2/W/ApplyAdam	ApplyAdamCNN_encoder_cat/Conv2D_2/W!CNN_encoder_cat/Conv2D_2/W/Adam_2!CNN_encoder_cat/Conv2D_2/W/Adam_3beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonKgradients_1/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
use_nesterov( 
ø
2Adam_1/update_CNN_encoder_cat/Conv2D_2/b/ApplyAdam	ApplyAdamCNN_encoder_cat/Conv2D_2/b!CNN_encoder_cat/Conv2D_2/b/Adam_2!CNN_encoder_cat/Conv2D_2/b/Adam_3beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonLgradients_1/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
use_nesterov( 
«
0Adam_1/update_CNN_encoder_cat/catout/W/ApplyAdam	ApplyAdamCNN_encoder_cat/catout/WCNN_encoder_cat/catout/W/Adam_2CNN_encoder_cat/catout/W/Adam_3beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonIgradients_1/CNN_encoder_cat/catout/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*+
_class!
loc:@CNN_encoder_cat/catout/W*
use_nesterov( 
¬
0Adam_1/update_CNN_encoder_cat/catout/b/ApplyAdam	ApplyAdamCNN_encoder_cat/catout/bCNN_encoder_cat/catout/b/Adam_2CNN_encoder_cat/catout/b/Adam_3beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonJgradients_1/CNN_encoder_cat/catout/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*+
_class!
loc:@CNN_encoder_cat/catout/b*
use_nesterov( 

.Adam_1/update_CNN_encoder_cat/zout/W/ApplyAdam	ApplyAdamCNN_encoder_cat/zout/WCNN_encoder_cat/zout/W/Adam_2CNN_encoder_cat/zout/W/Adam_3beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonGgradients_1/CNN_encoder_cat/zout/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@CNN_encoder_cat/zout/W*
use_nesterov( 
 
.Adam_1/update_CNN_encoder_cat/zout/b/ApplyAdam	ApplyAdamCNN_encoder_cat/zout/bCNN_encoder_cat/zout/b/Adam_2CNN_encoder_cat/zout/b/Adam_3beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonHgradients_1/CNN_encoder_cat/zout/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@CNN_encoder_cat/zout/b*
use_nesterov( 
Ė
(Adam_1/update_discriminator/w0/ApplyAdam	ApplyAdamdiscriminator/w0discriminator/w0/Adamdiscriminator/w0/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilongradients_1/AddN_14*
use_locking( *
T0*#
_class
loc:@discriminator/w0*
use_nesterov( 
Ė
(Adam_1/update_discriminator/b0/ApplyAdam	ApplyAdamdiscriminator/b0discriminator/b0/Adamdiscriminator/b0/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilongradients_1/AddN_12*
use_locking( *
T0*#
_class
loc:@discriminator/b0*
use_nesterov( 
Ė
(Adam_1/update_discriminator/w1/ApplyAdam	ApplyAdamdiscriminator/w1discriminator/w1/Adamdiscriminator/w1/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilongradients_1/AddN_10*
use_locking( *
T0*#
_class
loc:@discriminator/w1*
use_nesterov( 
Ź
(Adam_1/update_discriminator/b1/ApplyAdam	ApplyAdamdiscriminator/b1discriminator/b1/Adamdiscriminator/b1/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilongradients_1/AddN_8*
use_locking( *
T0*#
_class
loc:@discriminator/b1*
use_nesterov( 
Ź
(Adam_1/update_discriminator/wo/ApplyAdam	ApplyAdamdiscriminator/wodiscriminator/wo/Adamdiscriminator/wo/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilongradients_1/AddN_6*
use_locking( *
T0*#
_class
loc:@discriminator/wo*
use_nesterov( 
Ź
(Adam_1/update_discriminator/bo/ApplyAdam	ApplyAdamdiscriminator/bodiscriminator/bo/Adamdiscriminator/bo/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilongradients_1/AddN_4*
use_locking( *
T0*#
_class
loc:@discriminator/bo*
use_nesterov( 
ß
,Adam_1/update_discriminator_cat/w0/ApplyAdam	ApplyAdamdiscriminator_cat/w0discriminator_cat/w0/Adamdiscriminator_cat/w0/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilongradients_1/AddN_15*
use_locking( *
T0*'
_class
loc:@discriminator_cat/w0*
use_nesterov( 
ß
,Adam_1/update_discriminator_cat/b0/ApplyAdam	ApplyAdamdiscriminator_cat/b0discriminator_cat/b0/Adamdiscriminator_cat/b0/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilongradients_1/AddN_13*
use_locking( *
T0*'
_class
loc:@discriminator_cat/b0*
use_nesterov( 
ß
,Adam_1/update_discriminator_cat/w1/ApplyAdam	ApplyAdamdiscriminator_cat/w1discriminator_cat/w1/Adamdiscriminator_cat/w1/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilongradients_1/AddN_11*
use_locking( *
T0*'
_class
loc:@discriminator_cat/w1*
use_nesterov( 
Ž
,Adam_1/update_discriminator_cat/b1/ApplyAdam	ApplyAdamdiscriminator_cat/b1discriminator_cat/b1/Adamdiscriminator_cat/b1/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilongradients_1/AddN_9*
use_locking( *
T0*'
_class
loc:@discriminator_cat/b1*
use_nesterov( 
Ž
,Adam_1/update_discriminator_cat/wo/ApplyAdam	ApplyAdamdiscriminator_cat/wodiscriminator_cat/wo/Adamdiscriminator_cat/wo/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilongradients_1/AddN_7*
use_locking( *
T0*'
_class
loc:@discriminator_cat/wo*
use_nesterov( 
Ž
,Adam_1/update_discriminator_cat/bo/ApplyAdam	ApplyAdamdiscriminator_cat/bodiscriminator_cat/bo/Adamdiscriminator_cat/bo/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilongradients_1/AddN_5*
use_locking( *
T0*'
_class
loc:@discriminator_cat/bo*
use_nesterov( 
	

Adam_1/mulMulbeta1_power_1/readAdam_1/beta11^Adam_1/update_CNN_encoder_cat/Conv2D/W/ApplyAdam1^Adam_1/update_CNN_encoder_cat/Conv2D/b/ApplyAdam3^Adam_1/update_CNN_encoder_cat/Conv2D_1/W/ApplyAdam3^Adam_1/update_CNN_encoder_cat/Conv2D_1/b/ApplyAdam3^Adam_1/update_CNN_encoder_cat/Conv2D_2/W/ApplyAdam3^Adam_1/update_CNN_encoder_cat/Conv2D_2/b/ApplyAdam1^Adam_1/update_CNN_encoder_cat/catout/W/ApplyAdam1^Adam_1/update_CNN_encoder_cat/catout/b/ApplyAdam/^Adam_1/update_CNN_encoder_cat/zout/W/ApplyAdam/^Adam_1/update_CNN_encoder_cat/zout/b/ApplyAdam)^Adam_1/update_discriminator/b0/ApplyAdam)^Adam_1/update_discriminator/b1/ApplyAdam)^Adam_1/update_discriminator/bo/ApplyAdam)^Adam_1/update_discriminator/w0/ApplyAdam)^Adam_1/update_discriminator/w1/ApplyAdam)^Adam_1/update_discriminator/wo/ApplyAdam-^Adam_1/update_discriminator_cat/b0/ApplyAdam-^Adam_1/update_discriminator_cat/b1/ApplyAdam-^Adam_1/update_discriminator_cat/bo/ApplyAdam-^Adam_1/update_discriminator_cat/w0/ApplyAdam-^Adam_1/update_discriminator_cat/w1/ApplyAdam-^Adam_1/update_discriminator_cat/wo/ApplyAdam*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W

Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
use_locking( *
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(
	
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta21^Adam_1/update_CNN_encoder_cat/Conv2D/W/ApplyAdam1^Adam_1/update_CNN_encoder_cat/Conv2D/b/ApplyAdam3^Adam_1/update_CNN_encoder_cat/Conv2D_1/W/ApplyAdam3^Adam_1/update_CNN_encoder_cat/Conv2D_1/b/ApplyAdam3^Adam_1/update_CNN_encoder_cat/Conv2D_2/W/ApplyAdam3^Adam_1/update_CNN_encoder_cat/Conv2D_2/b/ApplyAdam1^Adam_1/update_CNN_encoder_cat/catout/W/ApplyAdam1^Adam_1/update_CNN_encoder_cat/catout/b/ApplyAdam/^Adam_1/update_CNN_encoder_cat/zout/W/ApplyAdam/^Adam_1/update_CNN_encoder_cat/zout/b/ApplyAdam)^Adam_1/update_discriminator/b0/ApplyAdam)^Adam_1/update_discriminator/b1/ApplyAdam)^Adam_1/update_discriminator/bo/ApplyAdam)^Adam_1/update_discriminator/w0/ApplyAdam)^Adam_1/update_discriminator/w1/ApplyAdam)^Adam_1/update_discriminator/wo/ApplyAdam-^Adam_1/update_discriminator_cat/b0/ApplyAdam-^Adam_1/update_discriminator_cat/b1/ApplyAdam-^Adam_1/update_discriminator_cat/bo/ApplyAdam-^Adam_1/update_discriminator_cat/w0/ApplyAdam-^Adam_1/update_discriminator_cat/w1/ApplyAdam-^Adam_1/update_discriminator_cat/wo/ApplyAdam*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W

Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
use_locking( *
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(
Ī
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_11^Adam_1/update_CNN_encoder_cat/Conv2D/W/ApplyAdam1^Adam_1/update_CNN_encoder_cat/Conv2D/b/ApplyAdam3^Adam_1/update_CNN_encoder_cat/Conv2D_1/W/ApplyAdam3^Adam_1/update_CNN_encoder_cat/Conv2D_1/b/ApplyAdam3^Adam_1/update_CNN_encoder_cat/Conv2D_2/W/ApplyAdam3^Adam_1/update_CNN_encoder_cat/Conv2D_2/b/ApplyAdam1^Adam_1/update_CNN_encoder_cat/catout/W/ApplyAdam1^Adam_1/update_CNN_encoder_cat/catout/b/ApplyAdam/^Adam_1/update_CNN_encoder_cat/zout/W/ApplyAdam/^Adam_1/update_CNN_encoder_cat/zout/b/ApplyAdam)^Adam_1/update_discriminator/b0/ApplyAdam)^Adam_1/update_discriminator/b1/ApplyAdam)^Adam_1/update_discriminator/bo/ApplyAdam)^Adam_1/update_discriminator/w0/ApplyAdam)^Adam_1/update_discriminator/w1/ApplyAdam)^Adam_1/update_discriminator/wo/ApplyAdam-^Adam_1/update_discriminator_cat/b0/ApplyAdam-^Adam_1/update_discriminator_cat/b1/ApplyAdam-^Adam_1/update_discriminator_cat/bo/ApplyAdam-^Adam_1/update_discriminator_cat/w0/ApplyAdam-^Adam_1/update_discriminator_cat/w1/ApplyAdam-^Adam_1/update_discriminator_cat/wo/ApplyAdam
:
gradients_2/ShapeConst*
valueB *
dtype0
B
gradients_2/grad_ys_0Const*
valueB
 *  ?*
dtype0
]
gradients_2/FillFillgradients_2/Shapegradients_2/grad_ys_0*
T0*

index_type0
N
%gradients_2/Mean_8_grad/Reshape/shapeConst*
valueB *
dtype0
z
gradients_2/Mean_8_grad/ReshapeReshapegradients_2/Fill%gradients_2/Mean_8_grad/Reshape/shape*
T0*
Tshape0
F
gradients_2/Mean_8_grad/ConstConst*
valueB *
dtype0

gradients_2/Mean_8_grad/TileTilegradients_2/Mean_8_grad/Reshapegradients_2/Mean_8_grad/Const*

Tmultiples0*
T0
L
gradients_2/Mean_8_grad/Const_1Const*
valueB
 *  ?*
dtype0
r
gradients_2/Mean_8_grad/truedivRealDivgradients_2/Mean_8_grad/Tilegradients_2/Mean_8_grad/Const_1*
T0
Q
'gradients_2/add_3_grad/tuple/group_depsNoOp ^gradients_2/Mean_8_grad/truediv
Ć
/gradients_2/add_3_grad/tuple/control_dependencyIdentitygradients_2/Mean_8_grad/truediv(^gradients_2/add_3_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_2/Mean_8_grad/truediv
Å
1gradients_2/add_3_grad/tuple/control_dependency_1Identitygradients_2/Mean_8_grad/truediv(^gradients_2/add_3_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_2/Mean_8_grad/truediv
Z
%gradients_2/Mean_3_grad/Reshape/shapeConst*
valueB"      *
dtype0

gradients_2/Mean_3_grad/ReshapeReshape/gradients_2/add_3_grad/tuple/control_dependency%gradients_2/Mean_3_grad/Reshape/shape*
T0*
Tshape0
P
gradients_2/Mean_3_grad/ShapeShapelogistic_loss_2*
T0*
out_type0

gradients_2/Mean_3_grad/TileTilegradients_2/Mean_3_grad/Reshapegradients_2/Mean_3_grad/Shape*

Tmultiples0*
T0
R
gradients_2/Mean_3_grad/Shape_1Shapelogistic_loss_2*
T0*
out_type0
H
gradients_2/Mean_3_grad/Shape_2Const*
valueB *
dtype0
K
gradients_2/Mean_3_grad/ConstConst*
valueB: *
dtype0

gradients_2/Mean_3_grad/ProdProdgradients_2/Mean_3_grad/Shape_1gradients_2/Mean_3_grad/Const*

Tidx0*
	keep_dims( *
T0
M
gradients_2/Mean_3_grad/Const_1Const*
valueB: *
dtype0

gradients_2/Mean_3_grad/Prod_1Prodgradients_2/Mean_3_grad/Shape_2gradients_2/Mean_3_grad/Const_1*

Tidx0*
	keep_dims( *
T0
K
!gradients_2/Mean_3_grad/Maximum/yConst*
value	B :*
dtype0
v
gradients_2/Mean_3_grad/MaximumMaximumgradients_2/Mean_3_grad/Prod_1!gradients_2/Mean_3_grad/Maximum/y*
T0
t
 gradients_2/Mean_3_grad/floordivFloorDivgradients_2/Mean_3_grad/Prodgradients_2/Mean_3_grad/Maximum*
T0
n
gradients_2/Mean_3_grad/CastCast gradients_2/Mean_3_grad/floordiv*

SrcT0*
Truncate( *

DstT0
o
gradients_2/Mean_3_grad/truedivRealDivgradients_2/Mean_3_grad/Tilegradients_2/Mean_3_grad/Cast*
T0
Z
%gradients_2/Mean_6_grad/Reshape/shapeConst*
valueB"      *
dtype0

gradients_2/Mean_6_grad/ReshapeReshape1gradients_2/add_3_grad/tuple/control_dependency_1%gradients_2/Mean_6_grad/Reshape/shape*
T0*
Tshape0
P
gradients_2/Mean_6_grad/ShapeShapelogistic_loss_5*
T0*
out_type0

gradients_2/Mean_6_grad/TileTilegradients_2/Mean_6_grad/Reshapegradients_2/Mean_6_grad/Shape*

Tmultiples0*
T0
R
gradients_2/Mean_6_grad/Shape_1Shapelogistic_loss_5*
T0*
out_type0
H
gradients_2/Mean_6_grad/Shape_2Const*
valueB *
dtype0
K
gradients_2/Mean_6_grad/ConstConst*
valueB: *
dtype0

gradients_2/Mean_6_grad/ProdProdgradients_2/Mean_6_grad/Shape_1gradients_2/Mean_6_grad/Const*

Tidx0*
	keep_dims( *
T0
M
gradients_2/Mean_6_grad/Const_1Const*
valueB: *
dtype0

gradients_2/Mean_6_grad/Prod_1Prodgradients_2/Mean_6_grad/Shape_2gradients_2/Mean_6_grad/Const_1*

Tidx0*
	keep_dims( *
T0
K
!gradients_2/Mean_6_grad/Maximum/yConst*
value	B :*
dtype0
v
gradients_2/Mean_6_grad/MaximumMaximumgradients_2/Mean_6_grad/Prod_1!gradients_2/Mean_6_grad/Maximum/y*
T0
t
 gradients_2/Mean_6_grad/floordivFloorDivgradients_2/Mean_6_grad/Prodgradients_2/Mean_6_grad/Maximum*
T0
n
gradients_2/Mean_6_grad/CastCast gradients_2/Mean_6_grad/floordiv*

SrcT0*
Truncate( *

DstT0
o
gradients_2/Mean_6_grad/truedivRealDivgradients_2/Mean_6_grad/Tilegradients_2/Mean_6_grad/Cast*
T0
]
&gradients_2/logistic_loss_2_grad/ShapeShapelogistic_loss_2/sub*
T0*
out_type0
a
(gradients_2/logistic_loss_2_grad/Shape_1Shapelogistic_loss_2/Log1p*
T0*
out_type0
Ŗ
6gradients_2/logistic_loss_2_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients_2/logistic_loss_2_grad/Shape(gradients_2/logistic_loss_2_grad/Shape_1*
T0
Ŗ
$gradients_2/logistic_loss_2_grad/SumSumgradients_2/Mean_3_grad/truediv6gradients_2/logistic_loss_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0

(gradients_2/logistic_loss_2_grad/ReshapeReshape$gradients_2/logistic_loss_2_grad/Sum&gradients_2/logistic_loss_2_grad/Shape*
T0*
Tshape0
®
&gradients_2/logistic_loss_2_grad/Sum_1Sumgradients_2/Mean_3_grad/truediv8gradients_2/logistic_loss_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0

*gradients_2/logistic_loss_2_grad/Reshape_1Reshape&gradients_2/logistic_loss_2_grad/Sum_1(gradients_2/logistic_loss_2_grad/Shape_1*
T0*
Tshape0

1gradients_2/logistic_loss_2_grad/tuple/group_depsNoOp)^gradients_2/logistic_loss_2_grad/Reshape+^gradients_2/logistic_loss_2_grad/Reshape_1
é
9gradients_2/logistic_loss_2_grad/tuple/control_dependencyIdentity(gradients_2/logistic_loss_2_grad/Reshape2^gradients_2/logistic_loss_2_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_2/logistic_loss_2_grad/Reshape
ļ
;gradients_2/logistic_loss_2_grad/tuple/control_dependency_1Identity*gradients_2/logistic_loss_2_grad/Reshape_12^gradients_2/logistic_loss_2_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_2/logistic_loss_2_grad/Reshape_1
]
&gradients_2/logistic_loss_5_grad/ShapeShapelogistic_loss_5/sub*
T0*
out_type0
a
(gradients_2/logistic_loss_5_grad/Shape_1Shapelogistic_loss_5/Log1p*
T0*
out_type0
Ŗ
6gradients_2/logistic_loss_5_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients_2/logistic_loss_5_grad/Shape(gradients_2/logistic_loss_5_grad/Shape_1*
T0
Ŗ
$gradients_2/logistic_loss_5_grad/SumSumgradients_2/Mean_6_grad/truediv6gradients_2/logistic_loss_5_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0

(gradients_2/logistic_loss_5_grad/ReshapeReshape$gradients_2/logistic_loss_5_grad/Sum&gradients_2/logistic_loss_5_grad/Shape*
T0*
Tshape0
®
&gradients_2/logistic_loss_5_grad/Sum_1Sumgradients_2/Mean_6_grad/truediv8gradients_2/logistic_loss_5_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0

*gradients_2/logistic_loss_5_grad/Reshape_1Reshape&gradients_2/logistic_loss_5_grad/Sum_1(gradients_2/logistic_loss_5_grad/Shape_1*
T0*
Tshape0

1gradients_2/logistic_loss_5_grad/tuple/group_depsNoOp)^gradients_2/logistic_loss_5_grad/Reshape+^gradients_2/logistic_loss_5_grad/Reshape_1
é
9gradients_2/logistic_loss_5_grad/tuple/control_dependencyIdentity(gradients_2/logistic_loss_5_grad/Reshape2^gradients_2/logistic_loss_5_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_2/logistic_loss_5_grad/Reshape
ļ
;gradients_2/logistic_loss_5_grad/tuple/control_dependency_1Identity*gradients_2/logistic_loss_5_grad/Reshape_12^gradients_2/logistic_loss_5_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_2/logistic_loss_5_grad/Reshape_1
d
*gradients_2/logistic_loss_2/sub_grad/ShapeShapelogistic_loss_2/Select*
T0*
out_type0
c
,gradients_2/logistic_loss_2/sub_grad/Shape_1Shapelogistic_loss_2/mul*
T0*
out_type0
¶
:gradients_2/logistic_loss_2/sub_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_2/logistic_loss_2/sub_grad/Shape,gradients_2/logistic_loss_2/sub_grad/Shape_1*
T0
Ģ
(gradients_2/logistic_loss_2/sub_grad/SumSum9gradients_2/logistic_loss_2_grad/tuple/control_dependency:gradients_2/logistic_loss_2/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
¤
,gradients_2/logistic_loss_2/sub_grad/ReshapeReshape(gradients_2/logistic_loss_2/sub_grad/Sum*gradients_2/logistic_loss_2/sub_grad/Shape*
T0*
Tshape0
s
(gradients_2/logistic_loss_2/sub_grad/NegNeg9gradients_2/logistic_loss_2_grad/tuple/control_dependency*
T0
æ
*gradients_2/logistic_loss_2/sub_grad/Sum_1Sum(gradients_2/logistic_loss_2/sub_grad/Neg<gradients_2/logistic_loss_2/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ŗ
.gradients_2/logistic_loss_2/sub_grad/Reshape_1Reshape*gradients_2/logistic_loss_2/sub_grad/Sum_1,gradients_2/logistic_loss_2/sub_grad/Shape_1*
T0*
Tshape0

5gradients_2/logistic_loss_2/sub_grad/tuple/group_depsNoOp-^gradients_2/logistic_loss_2/sub_grad/Reshape/^gradients_2/logistic_loss_2/sub_grad/Reshape_1
ł
=gradients_2/logistic_loss_2/sub_grad/tuple/control_dependencyIdentity,gradients_2/logistic_loss_2/sub_grad/Reshape6^gradients_2/logistic_loss_2/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_2/logistic_loss_2/sub_grad/Reshape
’
?gradients_2/logistic_loss_2/sub_grad/tuple/control_dependency_1Identity.gradients_2/logistic_loss_2/sub_grad/Reshape_16^gradients_2/logistic_loss_2/sub_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_2/logistic_loss_2/sub_grad/Reshape_1

,gradients_2/logistic_loss_2/Log1p_grad/add/xConst<^gradients_2/logistic_loss_2_grad/tuple/control_dependency_1*
valueB
 *  ?*
dtype0

*gradients_2/logistic_loss_2/Log1p_grad/addAddV2,gradients_2/logistic_loss_2/Log1p_grad/add/xlogistic_loss_2/Exp*
T0
t
1gradients_2/logistic_loss_2/Log1p_grad/Reciprocal
Reciprocal*gradients_2/logistic_loss_2/Log1p_grad/add*
T0
Ŗ
*gradients_2/logistic_loss_2/Log1p_grad/mulMul;gradients_2/logistic_loss_2_grad/tuple/control_dependency_11gradients_2/logistic_loss_2/Log1p_grad/Reciprocal*
T0
d
*gradients_2/logistic_loss_5/sub_grad/ShapeShapelogistic_loss_5/Select*
T0*
out_type0
c
,gradients_2/logistic_loss_5/sub_grad/Shape_1Shapelogistic_loss_5/mul*
T0*
out_type0
¶
:gradients_2/logistic_loss_5/sub_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_2/logistic_loss_5/sub_grad/Shape,gradients_2/logistic_loss_5/sub_grad/Shape_1*
T0
Ģ
(gradients_2/logistic_loss_5/sub_grad/SumSum9gradients_2/logistic_loss_5_grad/tuple/control_dependency:gradients_2/logistic_loss_5/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
¤
,gradients_2/logistic_loss_5/sub_grad/ReshapeReshape(gradients_2/logistic_loss_5/sub_grad/Sum*gradients_2/logistic_loss_5/sub_grad/Shape*
T0*
Tshape0
s
(gradients_2/logistic_loss_5/sub_grad/NegNeg9gradients_2/logistic_loss_5_grad/tuple/control_dependency*
T0
æ
*gradients_2/logistic_loss_5/sub_grad/Sum_1Sum(gradients_2/logistic_loss_5/sub_grad/Neg<gradients_2/logistic_loss_5/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ŗ
.gradients_2/logistic_loss_5/sub_grad/Reshape_1Reshape*gradients_2/logistic_loss_5/sub_grad/Sum_1,gradients_2/logistic_loss_5/sub_grad/Shape_1*
T0*
Tshape0

5gradients_2/logistic_loss_5/sub_grad/tuple/group_depsNoOp-^gradients_2/logistic_loss_5/sub_grad/Reshape/^gradients_2/logistic_loss_5/sub_grad/Reshape_1
ł
=gradients_2/logistic_loss_5/sub_grad/tuple/control_dependencyIdentity,gradients_2/logistic_loss_5/sub_grad/Reshape6^gradients_2/logistic_loss_5/sub_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_2/logistic_loss_5/sub_grad/Reshape
’
?gradients_2/logistic_loss_5/sub_grad/tuple/control_dependency_1Identity.gradients_2/logistic_loss_5/sub_grad/Reshape_16^gradients_2/logistic_loss_5/sub_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_2/logistic_loss_5/sub_grad/Reshape_1

,gradients_2/logistic_loss_5/Log1p_grad/add/xConst<^gradients_2/logistic_loss_5_grad/tuple/control_dependency_1*
valueB
 *  ?*
dtype0

*gradients_2/logistic_loss_5/Log1p_grad/addAddV2,gradients_2/logistic_loss_5/Log1p_grad/add/xlogistic_loss_5/Exp*
T0
t
1gradients_2/logistic_loss_5/Log1p_grad/Reciprocal
Reciprocal*gradients_2/logistic_loss_5/Log1p_grad/add*
T0
Ŗ
*gradients_2/logistic_loss_5/Log1p_grad/mulMul;gradients_2/logistic_loss_5_grad/tuple/control_dependency_11gradients_2/logistic_loss_5/Log1p_grad/Reciprocal*
T0
_
2gradients_2/logistic_loss_2/Select_grad/zeros_like	ZerosLikediscriminator_1/add_2*
T0
Ņ
.gradients_2/logistic_loss_2/Select_grad/SelectSelectlogistic_loss_2/GreaterEqual=gradients_2/logistic_loss_2/sub_grad/tuple/control_dependency2gradients_2/logistic_loss_2/Select_grad/zeros_like*
T0
Ō
0gradients_2/logistic_loss_2/Select_grad/Select_1Selectlogistic_loss_2/GreaterEqual2gradients_2/logistic_loss_2/Select_grad/zeros_like=gradients_2/logistic_loss_2/sub_grad/tuple/control_dependency*
T0
¤
8gradients_2/logistic_loss_2/Select_grad/tuple/group_depsNoOp/^gradients_2/logistic_loss_2/Select_grad/Select1^gradients_2/logistic_loss_2/Select_grad/Select_1

@gradients_2/logistic_loss_2/Select_grad/tuple/control_dependencyIdentity.gradients_2/logistic_loss_2/Select_grad/Select9^gradients_2/logistic_loss_2/Select_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_2/logistic_loss_2/Select_grad/Select

Bgradients_2/logistic_loss_2/Select_grad/tuple/control_dependency_1Identity0gradients_2/logistic_loss_2/Select_grad/Select_19^gradients_2/logistic_loss_2/Select_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_2/logistic_loss_2/Select_grad/Select_1
c
*gradients_2/logistic_loss_2/mul_grad/ShapeShapediscriminator_1/add_2*
T0*
out_type0
[
,gradients_2/logistic_loss_2/mul_grad/Shape_1Shapeones_like_1*
T0*
out_type0
¶
:gradients_2/logistic_loss_2/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_2/logistic_loss_2/mul_grad/Shape,gradients_2/logistic_loss_2/mul_grad/Shape_1*
T0

(gradients_2/logistic_loss_2/mul_grad/MulMul?gradients_2/logistic_loss_2/sub_grad/tuple/control_dependency_1ones_like_1*
T0
»
(gradients_2/logistic_loss_2/mul_grad/SumSum(gradients_2/logistic_loss_2/mul_grad/Mul:gradients_2/logistic_loss_2/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
¤
,gradients_2/logistic_loss_2/mul_grad/ReshapeReshape(gradients_2/logistic_loss_2/mul_grad/Sum*gradients_2/logistic_loss_2/mul_grad/Shape*
T0*
Tshape0

*gradients_2/logistic_loss_2/mul_grad/Mul_1Muldiscriminator_1/add_2?gradients_2/logistic_loss_2/sub_grad/tuple/control_dependency_1*
T0
Į
*gradients_2/logistic_loss_2/mul_grad/Sum_1Sum*gradients_2/logistic_loss_2/mul_grad/Mul_1<gradients_2/logistic_loss_2/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ŗ
.gradients_2/logistic_loss_2/mul_grad/Reshape_1Reshape*gradients_2/logistic_loss_2/mul_grad/Sum_1,gradients_2/logistic_loss_2/mul_grad/Shape_1*
T0*
Tshape0

5gradients_2/logistic_loss_2/mul_grad/tuple/group_depsNoOp-^gradients_2/logistic_loss_2/mul_grad/Reshape/^gradients_2/logistic_loss_2/mul_grad/Reshape_1
ł
=gradients_2/logistic_loss_2/mul_grad/tuple/control_dependencyIdentity,gradients_2/logistic_loss_2/mul_grad/Reshape6^gradients_2/logistic_loss_2/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_2/logistic_loss_2/mul_grad/Reshape
’
?gradients_2/logistic_loss_2/mul_grad/tuple/control_dependency_1Identity.gradients_2/logistic_loss_2/mul_grad/Reshape_16^gradients_2/logistic_loss_2/mul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_2/logistic_loss_2/mul_grad/Reshape_1
y
(gradients_2/logistic_loss_2/Exp_grad/mulMul*gradients_2/logistic_loss_2/Log1p_grad/mullogistic_loss_2/Exp*
T0
c
2gradients_2/logistic_loss_5/Select_grad/zeros_like	ZerosLikediscriminator_cat_1/add_2*
T0
Ņ
.gradients_2/logistic_loss_5/Select_grad/SelectSelectlogistic_loss_5/GreaterEqual=gradients_2/logistic_loss_5/sub_grad/tuple/control_dependency2gradients_2/logistic_loss_5/Select_grad/zeros_like*
T0
Ō
0gradients_2/logistic_loss_5/Select_grad/Select_1Selectlogistic_loss_5/GreaterEqual2gradients_2/logistic_loss_5/Select_grad/zeros_like=gradients_2/logistic_loss_5/sub_grad/tuple/control_dependency*
T0
¤
8gradients_2/logistic_loss_5/Select_grad/tuple/group_depsNoOp/^gradients_2/logistic_loss_5/Select_grad/Select1^gradients_2/logistic_loss_5/Select_grad/Select_1

@gradients_2/logistic_loss_5/Select_grad/tuple/control_dependencyIdentity.gradients_2/logistic_loss_5/Select_grad/Select9^gradients_2/logistic_loss_5/Select_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_2/logistic_loss_5/Select_grad/Select

Bgradients_2/logistic_loss_5/Select_grad/tuple/control_dependency_1Identity0gradients_2/logistic_loss_5/Select_grad/Select_19^gradients_2/logistic_loss_5/Select_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_2/logistic_loss_5/Select_grad/Select_1
g
*gradients_2/logistic_loss_5/mul_grad/ShapeShapediscriminator_cat_1/add_2*
T0*
out_type0
[
,gradients_2/logistic_loss_5/mul_grad/Shape_1Shapeones_like_3*
T0*
out_type0
¶
:gradients_2/logistic_loss_5/mul_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_2/logistic_loss_5/mul_grad/Shape,gradients_2/logistic_loss_5/mul_grad/Shape_1*
T0

(gradients_2/logistic_loss_5/mul_grad/MulMul?gradients_2/logistic_loss_5/sub_grad/tuple/control_dependency_1ones_like_3*
T0
»
(gradients_2/logistic_loss_5/mul_grad/SumSum(gradients_2/logistic_loss_5/mul_grad/Mul:gradients_2/logistic_loss_5/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
¤
,gradients_2/logistic_loss_5/mul_grad/ReshapeReshape(gradients_2/logistic_loss_5/mul_grad/Sum*gradients_2/logistic_loss_5/mul_grad/Shape*
T0*
Tshape0

*gradients_2/logistic_loss_5/mul_grad/Mul_1Muldiscriminator_cat_1/add_2?gradients_2/logistic_loss_5/sub_grad/tuple/control_dependency_1*
T0
Į
*gradients_2/logistic_loss_5/mul_grad/Sum_1Sum*gradients_2/logistic_loss_5/mul_grad/Mul_1<gradients_2/logistic_loss_5/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ŗ
.gradients_2/logistic_loss_5/mul_grad/Reshape_1Reshape*gradients_2/logistic_loss_5/mul_grad/Sum_1,gradients_2/logistic_loss_5/mul_grad/Shape_1*
T0*
Tshape0

5gradients_2/logistic_loss_5/mul_grad/tuple/group_depsNoOp-^gradients_2/logistic_loss_5/mul_grad/Reshape/^gradients_2/logistic_loss_5/mul_grad/Reshape_1
ł
=gradients_2/logistic_loss_5/mul_grad/tuple/control_dependencyIdentity,gradients_2/logistic_loss_5/mul_grad/Reshape6^gradients_2/logistic_loss_5/mul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_2/logistic_loss_5/mul_grad/Reshape
’
?gradients_2/logistic_loss_5/mul_grad/tuple/control_dependency_1Identity.gradients_2/logistic_loss_5/mul_grad/Reshape_16^gradients_2/logistic_loss_5/mul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_2/logistic_loss_5/mul_grad/Reshape_1
y
(gradients_2/logistic_loss_5/Exp_grad/mulMul*gradients_2/logistic_loss_5/Log1p_grad/mullogistic_loss_5/Exp*
T0
_
4gradients_2/logistic_loss_2/Select_1_grad/zeros_like	ZerosLikelogistic_loss_2/Neg*
T0
Į
0gradients_2/logistic_loss_2/Select_1_grad/SelectSelectlogistic_loss_2/GreaterEqual(gradients_2/logistic_loss_2/Exp_grad/mul4gradients_2/logistic_loss_2/Select_1_grad/zeros_like*
T0
Ć
2gradients_2/logistic_loss_2/Select_1_grad/Select_1Selectlogistic_loss_2/GreaterEqual4gradients_2/logistic_loss_2/Select_1_grad/zeros_like(gradients_2/logistic_loss_2/Exp_grad/mul*
T0
Ŗ
:gradients_2/logistic_loss_2/Select_1_grad/tuple/group_depsNoOp1^gradients_2/logistic_loss_2/Select_1_grad/Select3^gradients_2/logistic_loss_2/Select_1_grad/Select_1

Bgradients_2/logistic_loss_2/Select_1_grad/tuple/control_dependencyIdentity0gradients_2/logistic_loss_2/Select_1_grad/Select;^gradients_2/logistic_loss_2/Select_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_2/logistic_loss_2/Select_1_grad/Select

Dgradients_2/logistic_loss_2/Select_1_grad/tuple/control_dependency_1Identity2gradients_2/logistic_loss_2/Select_1_grad/Select_1;^gradients_2/logistic_loss_2/Select_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_2/logistic_loss_2/Select_1_grad/Select_1
_
4gradients_2/logistic_loss_5/Select_1_grad/zeros_like	ZerosLikelogistic_loss_5/Neg*
T0
Į
0gradients_2/logistic_loss_5/Select_1_grad/SelectSelectlogistic_loss_5/GreaterEqual(gradients_2/logistic_loss_5/Exp_grad/mul4gradients_2/logistic_loss_5/Select_1_grad/zeros_like*
T0
Ć
2gradients_2/logistic_loss_5/Select_1_grad/Select_1Selectlogistic_loss_5/GreaterEqual4gradients_2/logistic_loss_5/Select_1_grad/zeros_like(gradients_2/logistic_loss_5/Exp_grad/mul*
T0
Ŗ
:gradients_2/logistic_loss_5/Select_1_grad/tuple/group_depsNoOp1^gradients_2/logistic_loss_5/Select_1_grad/Select3^gradients_2/logistic_loss_5/Select_1_grad/Select_1

Bgradients_2/logistic_loss_5/Select_1_grad/tuple/control_dependencyIdentity0gradients_2/logistic_loss_5/Select_1_grad/Select;^gradients_2/logistic_loss_5/Select_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_2/logistic_loss_5/Select_1_grad/Select

Dgradients_2/logistic_loss_5/Select_1_grad/tuple/control_dependency_1Identity2gradients_2/logistic_loss_5/Select_1_grad/Select_1;^gradients_2/logistic_loss_5/Select_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_2/logistic_loss_5/Select_1_grad/Select_1
|
(gradients_2/logistic_loss_2/Neg_grad/NegNegBgradients_2/logistic_loss_2/Select_1_grad/tuple/control_dependency*
T0
|
(gradients_2/logistic_loss_5/Neg_grad/NegNegBgradients_2/logistic_loss_5/Select_1_grad/tuple/control_dependency*
T0
Ž
gradients_2/AddNAddN@gradients_2/logistic_loss_2/Select_grad/tuple/control_dependency=gradients_2/logistic_loss_2/mul_grad/tuple/control_dependencyDgradients_2/logistic_loss_2/Select_1_grad/tuple/control_dependency_1(gradients_2/logistic_loss_2/Neg_grad/Neg*
T0*A
_class7
53loc:@gradients_2/logistic_loss_2/Select_grad/Select*
N
h
,gradients_2/discriminator_1/add_2_grad/ShapeShapediscriminator_1/MatMul_2*
T0*
out_type0
g
.gradients_2/discriminator_1/add_2_grad/Shape_1Shapediscriminator/bo/read*
T0*
out_type0
¼
<gradients_2/discriminator_1/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients_2/discriminator_1/add_2_grad/Shape.gradients_2/discriminator_1/add_2_grad/Shape_1*
T0
§
*gradients_2/discriminator_1/add_2_grad/SumSumgradients_2/AddN<gradients_2/discriminator_1/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
Ŗ
.gradients_2/discriminator_1/add_2_grad/ReshapeReshape*gradients_2/discriminator_1/add_2_grad/Sum,gradients_2/discriminator_1/add_2_grad/Shape*
T0*
Tshape0
«
,gradients_2/discriminator_1/add_2_grad/Sum_1Sumgradients_2/AddN>gradients_2/discriminator_1/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
°
0gradients_2/discriminator_1/add_2_grad/Reshape_1Reshape,gradients_2/discriminator_1/add_2_grad/Sum_1.gradients_2/discriminator_1/add_2_grad/Shape_1*
T0*
Tshape0
£
7gradients_2/discriminator_1/add_2_grad/tuple/group_depsNoOp/^gradients_2/discriminator_1/add_2_grad/Reshape1^gradients_2/discriminator_1/add_2_grad/Reshape_1

?gradients_2/discriminator_1/add_2_grad/tuple/control_dependencyIdentity.gradients_2/discriminator_1/add_2_grad/Reshape8^gradients_2/discriminator_1/add_2_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_2/discriminator_1/add_2_grad/Reshape

Agradients_2/discriminator_1/add_2_grad/tuple/control_dependency_1Identity0gradients_2/discriminator_1/add_2_grad/Reshape_18^gradients_2/discriminator_1/add_2_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_2/discriminator_1/add_2_grad/Reshape_1
ą
gradients_2/AddN_1AddN@gradients_2/logistic_loss_5/Select_grad/tuple/control_dependency=gradients_2/logistic_loss_5/mul_grad/tuple/control_dependencyDgradients_2/logistic_loss_5/Select_1_grad/tuple/control_dependency_1(gradients_2/logistic_loss_5/Neg_grad/Neg*
T0*A
_class7
53loc:@gradients_2/logistic_loss_5/Select_grad/Select*
N
p
0gradients_2/discriminator_cat_1/add_2_grad/ShapeShapediscriminator_cat_1/MatMul_2*
T0*
out_type0
o
2gradients_2/discriminator_cat_1/add_2_grad/Shape_1Shapediscriminator_cat/bo/read*
T0*
out_type0
Č
@gradients_2/discriminator_cat_1/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs0gradients_2/discriminator_cat_1/add_2_grad/Shape2gradients_2/discriminator_cat_1/add_2_grad/Shape_1*
T0
±
.gradients_2/discriminator_cat_1/add_2_grad/SumSumgradients_2/AddN_1@gradients_2/discriminator_cat_1/add_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
¶
2gradients_2/discriminator_cat_1/add_2_grad/ReshapeReshape.gradients_2/discriminator_cat_1/add_2_grad/Sum0gradients_2/discriminator_cat_1/add_2_grad/Shape*
T0*
Tshape0
µ
0gradients_2/discriminator_cat_1/add_2_grad/Sum_1Sumgradients_2/AddN_1Bgradients_2/discriminator_cat_1/add_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
¼
4gradients_2/discriminator_cat_1/add_2_grad/Reshape_1Reshape0gradients_2/discriminator_cat_1/add_2_grad/Sum_12gradients_2/discriminator_cat_1/add_2_grad/Shape_1*
T0*
Tshape0
Æ
;gradients_2/discriminator_cat_1/add_2_grad/tuple/group_depsNoOp3^gradients_2/discriminator_cat_1/add_2_grad/Reshape5^gradients_2/discriminator_cat_1/add_2_grad/Reshape_1

Cgradients_2/discriminator_cat_1/add_2_grad/tuple/control_dependencyIdentity2gradients_2/discriminator_cat_1/add_2_grad/Reshape<^gradients_2/discriminator_cat_1/add_2_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_2/discriminator_cat_1/add_2_grad/Reshape

Egradients_2/discriminator_cat_1/add_2_grad/tuple/control_dependency_1Identity4gradients_2/discriminator_cat_1/add_2_grad/Reshape_1<^gradients_2/discriminator_cat_1/add_2_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_2/discriminator_cat_1/add_2_grad/Reshape_1
Į
0gradients_2/discriminator_1/MatMul_2_grad/MatMulMatMul?gradients_2/discriminator_1/add_2_grad/tuple/control_dependencydiscriminator/wo/read*
transpose_b(*
T0*
transpose_a( 
Ķ
2gradients_2/discriminator_1/MatMul_2_grad/MatMul_1MatMuldiscriminator_1/dropout_1/mul_1?gradients_2/discriminator_1/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
Ŗ
:gradients_2/discriminator_1/MatMul_2_grad/tuple/group_depsNoOp1^gradients_2/discriminator_1/MatMul_2_grad/MatMul3^gradients_2/discriminator_1/MatMul_2_grad/MatMul_1

Bgradients_2/discriminator_1/MatMul_2_grad/tuple/control_dependencyIdentity0gradients_2/discriminator_1/MatMul_2_grad/MatMul;^gradients_2/discriminator_1/MatMul_2_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_2/discriminator_1/MatMul_2_grad/MatMul

Dgradients_2/discriminator_1/MatMul_2_grad/tuple/control_dependency_1Identity2gradients_2/discriminator_1/MatMul_2_grad/MatMul_1;^gradients_2/discriminator_1/MatMul_2_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_2/discriminator_1/MatMul_2_grad/MatMul_1
Ķ
4gradients_2/discriminator_cat_1/MatMul_2_grad/MatMulMatMulCgradients_2/discriminator_cat_1/add_2_grad/tuple/control_dependencydiscriminator_cat/wo/read*
transpose_b(*
T0*
transpose_a( 
Ł
6gradients_2/discriminator_cat_1/MatMul_2_grad/MatMul_1MatMul#discriminator_cat_1/dropout_1/mul_1Cgradients_2/discriminator_cat_1/add_2_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
¶
>gradients_2/discriminator_cat_1/MatMul_2_grad/tuple/group_depsNoOp5^gradients_2/discriminator_cat_1/MatMul_2_grad/MatMul7^gradients_2/discriminator_cat_1/MatMul_2_grad/MatMul_1

Fgradients_2/discriminator_cat_1/MatMul_2_grad/tuple/control_dependencyIdentity4gradients_2/discriminator_cat_1/MatMul_2_grad/MatMul?^gradients_2/discriminator_cat_1/MatMul_2_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_2/discriminator_cat_1/MatMul_2_grad/MatMul
”
Hgradients_2/discriminator_cat_1/MatMul_2_grad/tuple/control_dependency_1Identity6gradients_2/discriminator_cat_1/MatMul_2_grad/MatMul_1?^gradients_2/discriminator_cat_1/MatMul_2_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients_2/discriminator_cat_1/MatMul_2_grad/MatMul_1
w
6gradients_2/discriminator_1/dropout_1/mul_1_grad/ShapeShapediscriminator_1/dropout_1/mul*
T0*
out_type0
z
8gradients_2/discriminator_1/dropout_1/mul_1_grad/Shape_1Shapediscriminator_1/dropout_1/Cast*
T0*
out_type0
Ś
Fgradients_2/discriminator_1/dropout_1/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_2/discriminator_1/dropout_1/mul_1_grad/Shape8gradients_2/discriminator_1/dropout_1/mul_1_grad/Shape_1*
T0
Ø
4gradients_2/discriminator_1/dropout_1/mul_1_grad/MulMulBgradients_2/discriminator_1/MatMul_2_grad/tuple/control_dependencydiscriminator_1/dropout_1/Cast*
T0
ß
4gradients_2/discriminator_1/dropout_1/mul_1_grad/SumSum4gradients_2/discriminator_1/dropout_1/mul_1_grad/MulFgradients_2/discriminator_1/dropout_1/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
Č
8gradients_2/discriminator_1/dropout_1/mul_1_grad/ReshapeReshape4gradients_2/discriminator_1/dropout_1/mul_1_grad/Sum6gradients_2/discriminator_1/dropout_1/mul_1_grad/Shape*
T0*
Tshape0
©
6gradients_2/discriminator_1/dropout_1/mul_1_grad/Mul_1Muldiscriminator_1/dropout_1/mulBgradients_2/discriminator_1/MatMul_2_grad/tuple/control_dependency*
T0
å
6gradients_2/discriminator_1/dropout_1/mul_1_grad/Sum_1Sum6gradients_2/discriminator_1/dropout_1/mul_1_grad/Mul_1Hgradients_2/discriminator_1/dropout_1/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ī
:gradients_2/discriminator_1/dropout_1/mul_1_grad/Reshape_1Reshape6gradients_2/discriminator_1/dropout_1/mul_1_grad/Sum_18gradients_2/discriminator_1/dropout_1/mul_1_grad/Shape_1*
T0*
Tshape0
Į
Agradients_2/discriminator_1/dropout_1/mul_1_grad/tuple/group_depsNoOp9^gradients_2/discriminator_1/dropout_1/mul_1_grad/Reshape;^gradients_2/discriminator_1/dropout_1/mul_1_grad/Reshape_1
©
Igradients_2/discriminator_1/dropout_1/mul_1_grad/tuple/control_dependencyIdentity8gradients_2/discriminator_1/dropout_1/mul_1_grad/ReshapeB^gradients_2/discriminator_1/dropout_1/mul_1_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_2/discriminator_1/dropout_1/mul_1_grad/Reshape
Æ
Kgradients_2/discriminator_1/dropout_1/mul_1_grad/tuple/control_dependency_1Identity:gradients_2/discriminator_1/dropout_1/mul_1_grad/Reshape_1B^gradients_2/discriminator_1/dropout_1/mul_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_2/discriminator_1/dropout_1/mul_1_grad/Reshape_1

:gradients_2/discriminator_cat_1/dropout_1/mul_1_grad/ShapeShape!discriminator_cat_1/dropout_1/mul*
T0*
out_type0

<gradients_2/discriminator_cat_1/dropout_1/mul_1_grad/Shape_1Shape"discriminator_cat_1/dropout_1/Cast*
T0*
out_type0
ę
Jgradients_2/discriminator_cat_1/dropout_1/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients_2/discriminator_cat_1/dropout_1/mul_1_grad/Shape<gradients_2/discriminator_cat_1/dropout_1/mul_1_grad/Shape_1*
T0
“
8gradients_2/discriminator_cat_1/dropout_1/mul_1_grad/MulMulFgradients_2/discriminator_cat_1/MatMul_2_grad/tuple/control_dependency"discriminator_cat_1/dropout_1/Cast*
T0
ė
8gradients_2/discriminator_cat_1/dropout_1/mul_1_grad/SumSum8gradients_2/discriminator_cat_1/dropout_1/mul_1_grad/MulJgradients_2/discriminator_cat_1/dropout_1/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
Ō
<gradients_2/discriminator_cat_1/dropout_1/mul_1_grad/ReshapeReshape8gradients_2/discriminator_cat_1/dropout_1/mul_1_grad/Sum:gradients_2/discriminator_cat_1/dropout_1/mul_1_grad/Shape*
T0*
Tshape0
µ
:gradients_2/discriminator_cat_1/dropout_1/mul_1_grad/Mul_1Mul!discriminator_cat_1/dropout_1/mulFgradients_2/discriminator_cat_1/MatMul_2_grad/tuple/control_dependency*
T0
ń
:gradients_2/discriminator_cat_1/dropout_1/mul_1_grad/Sum_1Sum:gradients_2/discriminator_cat_1/dropout_1/mul_1_grad/Mul_1Lgradients_2/discriminator_cat_1/dropout_1/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ś
>gradients_2/discriminator_cat_1/dropout_1/mul_1_grad/Reshape_1Reshape:gradients_2/discriminator_cat_1/dropout_1/mul_1_grad/Sum_1<gradients_2/discriminator_cat_1/dropout_1/mul_1_grad/Shape_1*
T0*
Tshape0
Ķ
Egradients_2/discriminator_cat_1/dropout_1/mul_1_grad/tuple/group_depsNoOp=^gradients_2/discriminator_cat_1/dropout_1/mul_1_grad/Reshape?^gradients_2/discriminator_cat_1/dropout_1/mul_1_grad/Reshape_1
¹
Mgradients_2/discriminator_cat_1/dropout_1/mul_1_grad/tuple/control_dependencyIdentity<gradients_2/discriminator_cat_1/dropout_1/mul_1_grad/ReshapeF^gradients_2/discriminator_cat_1/dropout_1/mul_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_2/discriminator_cat_1/dropout_1/mul_1_grad/Reshape
æ
Ogradients_2/discriminator_cat_1/dropout_1/mul_1_grad/tuple/control_dependency_1Identity>gradients_2/discriminator_cat_1/dropout_1/mul_1_grad/Reshape_1F^gradients_2/discriminator_cat_1/dropout_1/mul_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients_2/discriminator_cat_1/dropout_1/mul_1_grad/Reshape_1
n
4gradients_2/discriminator_1/dropout_1/mul_grad/ShapeShapediscriminator_1/Relu_1*
T0*
out_type0
{
6gradients_2/discriminator_1/dropout_1/mul_grad/Shape_1Shape!discriminator_1/dropout_1/truediv*
T0*
out_type0
Ō
Dgradients_2/discriminator_1/dropout_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients_2/discriminator_1/dropout_1/mul_grad/Shape6gradients_2/discriminator_1/dropout_1/mul_grad/Shape_1*
T0
°
2gradients_2/discriminator_1/dropout_1/mul_grad/MulMulIgradients_2/discriminator_1/dropout_1/mul_1_grad/tuple/control_dependency!discriminator_1/dropout_1/truediv*
T0
Ł
2gradients_2/discriminator_1/dropout_1/mul_grad/SumSum2gradients_2/discriminator_1/dropout_1/mul_grad/MulDgradients_2/discriminator_1/dropout_1/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
Ā
6gradients_2/discriminator_1/dropout_1/mul_grad/ReshapeReshape2gradients_2/discriminator_1/dropout_1/mul_grad/Sum4gradients_2/discriminator_1/dropout_1/mul_grad/Shape*
T0*
Tshape0
§
4gradients_2/discriminator_1/dropout_1/mul_grad/Mul_1Muldiscriminator_1/Relu_1Igradients_2/discriminator_1/dropout_1/mul_1_grad/tuple/control_dependency*
T0
ß
4gradients_2/discriminator_1/dropout_1/mul_grad/Sum_1Sum4gradients_2/discriminator_1/dropout_1/mul_grad/Mul_1Fgradients_2/discriminator_1/dropout_1/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Č
8gradients_2/discriminator_1/dropout_1/mul_grad/Reshape_1Reshape4gradients_2/discriminator_1/dropout_1/mul_grad/Sum_16gradients_2/discriminator_1/dropout_1/mul_grad/Shape_1*
T0*
Tshape0
»
?gradients_2/discriminator_1/dropout_1/mul_grad/tuple/group_depsNoOp7^gradients_2/discriminator_1/dropout_1/mul_grad/Reshape9^gradients_2/discriminator_1/dropout_1/mul_grad/Reshape_1
”
Ggradients_2/discriminator_1/dropout_1/mul_grad/tuple/control_dependencyIdentity6gradients_2/discriminator_1/dropout_1/mul_grad/Reshape@^gradients_2/discriminator_1/dropout_1/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients_2/discriminator_1/dropout_1/mul_grad/Reshape
§
Igradients_2/discriminator_1/dropout_1/mul_grad/tuple/control_dependency_1Identity8gradients_2/discriminator_1/dropout_1/mul_grad/Reshape_1@^gradients_2/discriminator_1/dropout_1/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_2/discriminator_1/dropout_1/mul_grad/Reshape_1
v
8gradients_2/discriminator_cat_1/dropout_1/mul_grad/ShapeShapediscriminator_cat_1/Relu_1*
T0*
out_type0

:gradients_2/discriminator_cat_1/dropout_1/mul_grad/Shape_1Shape%discriminator_cat_1/dropout_1/truediv*
T0*
out_type0
ą
Hgradients_2/discriminator_cat_1/dropout_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients_2/discriminator_cat_1/dropout_1/mul_grad/Shape:gradients_2/discriminator_cat_1/dropout_1/mul_grad/Shape_1*
T0
¼
6gradients_2/discriminator_cat_1/dropout_1/mul_grad/MulMulMgradients_2/discriminator_cat_1/dropout_1/mul_1_grad/tuple/control_dependency%discriminator_cat_1/dropout_1/truediv*
T0
å
6gradients_2/discriminator_cat_1/dropout_1/mul_grad/SumSum6gradients_2/discriminator_cat_1/dropout_1/mul_grad/MulHgradients_2/discriminator_cat_1/dropout_1/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
Ī
:gradients_2/discriminator_cat_1/dropout_1/mul_grad/ReshapeReshape6gradients_2/discriminator_cat_1/dropout_1/mul_grad/Sum8gradients_2/discriminator_cat_1/dropout_1/mul_grad/Shape*
T0*
Tshape0
³
8gradients_2/discriminator_cat_1/dropout_1/mul_grad/Mul_1Muldiscriminator_cat_1/Relu_1Mgradients_2/discriminator_cat_1/dropout_1/mul_1_grad/tuple/control_dependency*
T0
ė
8gradients_2/discriminator_cat_1/dropout_1/mul_grad/Sum_1Sum8gradients_2/discriminator_cat_1/dropout_1/mul_grad/Mul_1Jgradients_2/discriminator_cat_1/dropout_1/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ō
<gradients_2/discriminator_cat_1/dropout_1/mul_grad/Reshape_1Reshape8gradients_2/discriminator_cat_1/dropout_1/mul_grad/Sum_1:gradients_2/discriminator_cat_1/dropout_1/mul_grad/Shape_1*
T0*
Tshape0
Ē
Cgradients_2/discriminator_cat_1/dropout_1/mul_grad/tuple/group_depsNoOp;^gradients_2/discriminator_cat_1/dropout_1/mul_grad/Reshape=^gradients_2/discriminator_cat_1/dropout_1/mul_grad/Reshape_1
±
Kgradients_2/discriminator_cat_1/dropout_1/mul_grad/tuple/control_dependencyIdentity:gradients_2/discriminator_cat_1/dropout_1/mul_grad/ReshapeD^gradients_2/discriminator_cat_1/dropout_1/mul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_2/discriminator_cat_1/dropout_1/mul_grad/Reshape
·
Mgradients_2/discriminator_cat_1/dropout_1/mul_grad/tuple/control_dependency_1Identity<gradients_2/discriminator_cat_1/dropout_1/mul_grad/Reshape_1D^gradients_2/discriminator_cat_1/dropout_1/mul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_2/discriminator_cat_1/dropout_1/mul_grad/Reshape_1
¦
0gradients_2/discriminator_1/Relu_1_grad/ReluGradReluGradGgradients_2/discriminator_1/dropout_1/mul_grad/tuple/control_dependencydiscriminator_1/Relu_1*
T0
²
4gradients_2/discriminator_cat_1/Relu_1_grad/ReluGradReluGradKgradients_2/discriminator_cat_1/dropout_1/mul_grad/tuple/control_dependencydiscriminator_cat_1/Relu_1*
T0
h
,gradients_2/discriminator_1/add_1_grad/ShapeShapediscriminator_1/MatMul_1*
T0*
out_type0
g
.gradients_2/discriminator_1/add_1_grad/Shape_1Shapediscriminator/b1/read*
T0*
out_type0
¼
<gradients_2/discriminator_1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients_2/discriminator_1/add_1_grad/Shape.gradients_2/discriminator_1/add_1_grad/Shape_1*
T0
Ē
*gradients_2/discriminator_1/add_1_grad/SumSum0gradients_2/discriminator_1/Relu_1_grad/ReluGrad<gradients_2/discriminator_1/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
Ŗ
.gradients_2/discriminator_1/add_1_grad/ReshapeReshape*gradients_2/discriminator_1/add_1_grad/Sum,gradients_2/discriminator_1/add_1_grad/Shape*
T0*
Tshape0
Ė
,gradients_2/discriminator_1/add_1_grad/Sum_1Sum0gradients_2/discriminator_1/Relu_1_grad/ReluGrad>gradients_2/discriminator_1/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
°
0gradients_2/discriminator_1/add_1_grad/Reshape_1Reshape,gradients_2/discriminator_1/add_1_grad/Sum_1.gradients_2/discriminator_1/add_1_grad/Shape_1*
T0*
Tshape0
£
7gradients_2/discriminator_1/add_1_grad/tuple/group_depsNoOp/^gradients_2/discriminator_1/add_1_grad/Reshape1^gradients_2/discriminator_1/add_1_grad/Reshape_1

?gradients_2/discriminator_1/add_1_grad/tuple/control_dependencyIdentity.gradients_2/discriminator_1/add_1_grad/Reshape8^gradients_2/discriminator_1/add_1_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_2/discriminator_1/add_1_grad/Reshape

Agradients_2/discriminator_1/add_1_grad/tuple/control_dependency_1Identity0gradients_2/discriminator_1/add_1_grad/Reshape_18^gradients_2/discriminator_1/add_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_2/discriminator_1/add_1_grad/Reshape_1
p
0gradients_2/discriminator_cat_1/add_1_grad/ShapeShapediscriminator_cat_1/MatMul_1*
T0*
out_type0
o
2gradients_2/discriminator_cat_1/add_1_grad/Shape_1Shapediscriminator_cat/b1/read*
T0*
out_type0
Č
@gradients_2/discriminator_cat_1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs0gradients_2/discriminator_cat_1/add_1_grad/Shape2gradients_2/discriminator_cat_1/add_1_grad/Shape_1*
T0
Ó
.gradients_2/discriminator_cat_1/add_1_grad/SumSum4gradients_2/discriminator_cat_1/Relu_1_grad/ReluGrad@gradients_2/discriminator_cat_1/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
¶
2gradients_2/discriminator_cat_1/add_1_grad/ReshapeReshape.gradients_2/discriminator_cat_1/add_1_grad/Sum0gradients_2/discriminator_cat_1/add_1_grad/Shape*
T0*
Tshape0
×
0gradients_2/discriminator_cat_1/add_1_grad/Sum_1Sum4gradients_2/discriminator_cat_1/Relu_1_grad/ReluGradBgradients_2/discriminator_cat_1/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
¼
4gradients_2/discriminator_cat_1/add_1_grad/Reshape_1Reshape0gradients_2/discriminator_cat_1/add_1_grad/Sum_12gradients_2/discriminator_cat_1/add_1_grad/Shape_1*
T0*
Tshape0
Æ
;gradients_2/discriminator_cat_1/add_1_grad/tuple/group_depsNoOp3^gradients_2/discriminator_cat_1/add_1_grad/Reshape5^gradients_2/discriminator_cat_1/add_1_grad/Reshape_1

Cgradients_2/discriminator_cat_1/add_1_grad/tuple/control_dependencyIdentity2gradients_2/discriminator_cat_1/add_1_grad/Reshape<^gradients_2/discriminator_cat_1/add_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_2/discriminator_cat_1/add_1_grad/Reshape

Egradients_2/discriminator_cat_1/add_1_grad/tuple/control_dependency_1Identity4gradients_2/discriminator_cat_1/add_1_grad/Reshape_1<^gradients_2/discriminator_cat_1/add_1_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_2/discriminator_cat_1/add_1_grad/Reshape_1
Į
0gradients_2/discriminator_1/MatMul_1_grad/MatMulMatMul?gradients_2/discriminator_1/add_1_grad/tuple/control_dependencydiscriminator/w1/read*
transpose_b(*
T0*
transpose_a( 
Ė
2gradients_2/discriminator_1/MatMul_1_grad/MatMul_1MatMuldiscriminator_1/dropout/mul_1?gradients_2/discriminator_1/add_1_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
Ŗ
:gradients_2/discriminator_1/MatMul_1_grad/tuple/group_depsNoOp1^gradients_2/discriminator_1/MatMul_1_grad/MatMul3^gradients_2/discriminator_1/MatMul_1_grad/MatMul_1

Bgradients_2/discriminator_1/MatMul_1_grad/tuple/control_dependencyIdentity0gradients_2/discriminator_1/MatMul_1_grad/MatMul;^gradients_2/discriminator_1/MatMul_1_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_2/discriminator_1/MatMul_1_grad/MatMul

Dgradients_2/discriminator_1/MatMul_1_grad/tuple/control_dependency_1Identity2gradients_2/discriminator_1/MatMul_1_grad/MatMul_1;^gradients_2/discriminator_1/MatMul_1_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_2/discriminator_1/MatMul_1_grad/MatMul_1
Ķ
4gradients_2/discriminator_cat_1/MatMul_1_grad/MatMulMatMulCgradients_2/discriminator_cat_1/add_1_grad/tuple/control_dependencydiscriminator_cat/w1/read*
transpose_b(*
T0*
transpose_a( 
×
6gradients_2/discriminator_cat_1/MatMul_1_grad/MatMul_1MatMul!discriminator_cat_1/dropout/mul_1Cgradients_2/discriminator_cat_1/add_1_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
¶
>gradients_2/discriminator_cat_1/MatMul_1_grad/tuple/group_depsNoOp5^gradients_2/discriminator_cat_1/MatMul_1_grad/MatMul7^gradients_2/discriminator_cat_1/MatMul_1_grad/MatMul_1

Fgradients_2/discriminator_cat_1/MatMul_1_grad/tuple/control_dependencyIdentity4gradients_2/discriminator_cat_1/MatMul_1_grad/MatMul?^gradients_2/discriminator_cat_1/MatMul_1_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_2/discriminator_cat_1/MatMul_1_grad/MatMul
”
Hgradients_2/discriminator_cat_1/MatMul_1_grad/tuple/control_dependency_1Identity6gradients_2/discriminator_cat_1/MatMul_1_grad/MatMul_1?^gradients_2/discriminator_cat_1/MatMul_1_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients_2/discriminator_cat_1/MatMul_1_grad/MatMul_1
s
4gradients_2/discriminator_1/dropout/mul_1_grad/ShapeShapediscriminator_1/dropout/mul*
T0*
out_type0
v
6gradients_2/discriminator_1/dropout/mul_1_grad/Shape_1Shapediscriminator_1/dropout/Cast*
T0*
out_type0
Ō
Dgradients_2/discriminator_1/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients_2/discriminator_1/dropout/mul_1_grad/Shape6gradients_2/discriminator_1/dropout/mul_1_grad/Shape_1*
T0
¤
2gradients_2/discriminator_1/dropout/mul_1_grad/MulMulBgradients_2/discriminator_1/MatMul_1_grad/tuple/control_dependencydiscriminator_1/dropout/Cast*
T0
Ł
2gradients_2/discriminator_1/dropout/mul_1_grad/SumSum2gradients_2/discriminator_1/dropout/mul_1_grad/MulDgradients_2/discriminator_1/dropout/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
Ā
6gradients_2/discriminator_1/dropout/mul_1_grad/ReshapeReshape2gradients_2/discriminator_1/dropout/mul_1_grad/Sum4gradients_2/discriminator_1/dropout/mul_1_grad/Shape*
T0*
Tshape0
„
4gradients_2/discriminator_1/dropout/mul_1_grad/Mul_1Muldiscriminator_1/dropout/mulBgradients_2/discriminator_1/MatMul_1_grad/tuple/control_dependency*
T0
ß
4gradients_2/discriminator_1/dropout/mul_1_grad/Sum_1Sum4gradients_2/discriminator_1/dropout/mul_1_grad/Mul_1Fgradients_2/discriminator_1/dropout/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Č
8gradients_2/discriminator_1/dropout/mul_1_grad/Reshape_1Reshape4gradients_2/discriminator_1/dropout/mul_1_grad/Sum_16gradients_2/discriminator_1/dropout/mul_1_grad/Shape_1*
T0*
Tshape0
»
?gradients_2/discriminator_1/dropout/mul_1_grad/tuple/group_depsNoOp7^gradients_2/discriminator_1/dropout/mul_1_grad/Reshape9^gradients_2/discriminator_1/dropout/mul_1_grad/Reshape_1
”
Ggradients_2/discriminator_1/dropout/mul_1_grad/tuple/control_dependencyIdentity6gradients_2/discriminator_1/dropout/mul_1_grad/Reshape@^gradients_2/discriminator_1/dropout/mul_1_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients_2/discriminator_1/dropout/mul_1_grad/Reshape
§
Igradients_2/discriminator_1/dropout/mul_1_grad/tuple/control_dependency_1Identity8gradients_2/discriminator_1/dropout/mul_1_grad/Reshape_1@^gradients_2/discriminator_1/dropout/mul_1_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_2/discriminator_1/dropout/mul_1_grad/Reshape_1
{
8gradients_2/discriminator_cat_1/dropout/mul_1_grad/ShapeShapediscriminator_cat_1/dropout/mul*
T0*
out_type0
~
:gradients_2/discriminator_cat_1/dropout/mul_1_grad/Shape_1Shape discriminator_cat_1/dropout/Cast*
T0*
out_type0
ą
Hgradients_2/discriminator_cat_1/dropout/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients_2/discriminator_cat_1/dropout/mul_1_grad/Shape:gradients_2/discriminator_cat_1/dropout/mul_1_grad/Shape_1*
T0
°
6gradients_2/discriminator_cat_1/dropout/mul_1_grad/MulMulFgradients_2/discriminator_cat_1/MatMul_1_grad/tuple/control_dependency discriminator_cat_1/dropout/Cast*
T0
å
6gradients_2/discriminator_cat_1/dropout/mul_1_grad/SumSum6gradients_2/discriminator_cat_1/dropout/mul_1_grad/MulHgradients_2/discriminator_cat_1/dropout/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
Ī
:gradients_2/discriminator_cat_1/dropout/mul_1_grad/ReshapeReshape6gradients_2/discriminator_cat_1/dropout/mul_1_grad/Sum8gradients_2/discriminator_cat_1/dropout/mul_1_grad/Shape*
T0*
Tshape0
±
8gradients_2/discriminator_cat_1/dropout/mul_1_grad/Mul_1Muldiscriminator_cat_1/dropout/mulFgradients_2/discriminator_cat_1/MatMul_1_grad/tuple/control_dependency*
T0
ė
8gradients_2/discriminator_cat_1/dropout/mul_1_grad/Sum_1Sum8gradients_2/discriminator_cat_1/dropout/mul_1_grad/Mul_1Jgradients_2/discriminator_cat_1/dropout/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ō
<gradients_2/discriminator_cat_1/dropout/mul_1_grad/Reshape_1Reshape8gradients_2/discriminator_cat_1/dropout/mul_1_grad/Sum_1:gradients_2/discriminator_cat_1/dropout/mul_1_grad/Shape_1*
T0*
Tshape0
Ē
Cgradients_2/discriminator_cat_1/dropout/mul_1_grad/tuple/group_depsNoOp;^gradients_2/discriminator_cat_1/dropout/mul_1_grad/Reshape=^gradients_2/discriminator_cat_1/dropout/mul_1_grad/Reshape_1
±
Kgradients_2/discriminator_cat_1/dropout/mul_1_grad/tuple/control_dependencyIdentity:gradients_2/discriminator_cat_1/dropout/mul_1_grad/ReshapeD^gradients_2/discriminator_cat_1/dropout/mul_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_2/discriminator_cat_1/dropout/mul_1_grad/Reshape
·
Mgradients_2/discriminator_cat_1/dropout/mul_1_grad/tuple/control_dependency_1Identity<gradients_2/discriminator_cat_1/dropout/mul_1_grad/Reshape_1D^gradients_2/discriminator_cat_1/dropout/mul_1_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_2/discriminator_cat_1/dropout/mul_1_grad/Reshape_1
j
2gradients_2/discriminator_1/dropout/mul_grad/ShapeShapediscriminator_1/Relu*
T0*
out_type0
w
4gradients_2/discriminator_1/dropout/mul_grad/Shape_1Shapediscriminator_1/dropout/truediv*
T0*
out_type0
Ī
Bgradients_2/discriminator_1/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs2gradients_2/discriminator_1/dropout/mul_grad/Shape4gradients_2/discriminator_1/dropout/mul_grad/Shape_1*
T0
Ŗ
0gradients_2/discriminator_1/dropout/mul_grad/MulMulGgradients_2/discriminator_1/dropout/mul_1_grad/tuple/control_dependencydiscriminator_1/dropout/truediv*
T0
Ó
0gradients_2/discriminator_1/dropout/mul_grad/SumSum0gradients_2/discriminator_1/dropout/mul_grad/MulBgradients_2/discriminator_1/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
¼
4gradients_2/discriminator_1/dropout/mul_grad/ReshapeReshape0gradients_2/discriminator_1/dropout/mul_grad/Sum2gradients_2/discriminator_1/dropout/mul_grad/Shape*
T0*
Tshape0
”
2gradients_2/discriminator_1/dropout/mul_grad/Mul_1Muldiscriminator_1/ReluGgradients_2/discriminator_1/dropout/mul_1_grad/tuple/control_dependency*
T0
Ł
2gradients_2/discriminator_1/dropout/mul_grad/Sum_1Sum2gradients_2/discriminator_1/dropout/mul_grad/Mul_1Dgradients_2/discriminator_1/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ā
6gradients_2/discriminator_1/dropout/mul_grad/Reshape_1Reshape2gradients_2/discriminator_1/dropout/mul_grad/Sum_14gradients_2/discriminator_1/dropout/mul_grad/Shape_1*
T0*
Tshape0
µ
=gradients_2/discriminator_1/dropout/mul_grad/tuple/group_depsNoOp5^gradients_2/discriminator_1/dropout/mul_grad/Reshape7^gradients_2/discriminator_1/dropout/mul_grad/Reshape_1

Egradients_2/discriminator_1/dropout/mul_grad/tuple/control_dependencyIdentity4gradients_2/discriminator_1/dropout/mul_grad/Reshape>^gradients_2/discriminator_1/dropout/mul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_2/discriminator_1/dropout/mul_grad/Reshape

Ggradients_2/discriminator_1/dropout/mul_grad/tuple/control_dependency_1Identity6gradients_2/discriminator_1/dropout/mul_grad/Reshape_1>^gradients_2/discriminator_1/dropout/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients_2/discriminator_1/dropout/mul_grad/Reshape_1
r
6gradients_2/discriminator_cat_1/dropout/mul_grad/ShapeShapediscriminator_cat_1/Relu*
T0*
out_type0

8gradients_2/discriminator_cat_1/dropout/mul_grad/Shape_1Shape#discriminator_cat_1/dropout/truediv*
T0*
out_type0
Ś
Fgradients_2/discriminator_cat_1/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_2/discriminator_cat_1/dropout/mul_grad/Shape8gradients_2/discriminator_cat_1/dropout/mul_grad/Shape_1*
T0
¶
4gradients_2/discriminator_cat_1/dropout/mul_grad/MulMulKgradients_2/discriminator_cat_1/dropout/mul_1_grad/tuple/control_dependency#discriminator_cat_1/dropout/truediv*
T0
ß
4gradients_2/discriminator_cat_1/dropout/mul_grad/SumSum4gradients_2/discriminator_cat_1/dropout/mul_grad/MulFgradients_2/discriminator_cat_1/dropout/mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
Č
8gradients_2/discriminator_cat_1/dropout/mul_grad/ReshapeReshape4gradients_2/discriminator_cat_1/dropout/mul_grad/Sum6gradients_2/discriminator_cat_1/dropout/mul_grad/Shape*
T0*
Tshape0
­
6gradients_2/discriminator_cat_1/dropout/mul_grad/Mul_1Muldiscriminator_cat_1/ReluKgradients_2/discriminator_cat_1/dropout/mul_1_grad/tuple/control_dependency*
T0
å
6gradients_2/discriminator_cat_1/dropout/mul_grad/Sum_1Sum6gradients_2/discriminator_cat_1/dropout/mul_grad/Mul_1Hgradients_2/discriminator_cat_1/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ī
:gradients_2/discriminator_cat_1/dropout/mul_grad/Reshape_1Reshape6gradients_2/discriminator_cat_1/dropout/mul_grad/Sum_18gradients_2/discriminator_cat_1/dropout/mul_grad/Shape_1*
T0*
Tshape0
Į
Agradients_2/discriminator_cat_1/dropout/mul_grad/tuple/group_depsNoOp9^gradients_2/discriminator_cat_1/dropout/mul_grad/Reshape;^gradients_2/discriminator_cat_1/dropout/mul_grad/Reshape_1
©
Igradients_2/discriminator_cat_1/dropout/mul_grad/tuple/control_dependencyIdentity8gradients_2/discriminator_cat_1/dropout/mul_grad/ReshapeB^gradients_2/discriminator_cat_1/dropout/mul_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_2/discriminator_cat_1/dropout/mul_grad/Reshape
Æ
Kgradients_2/discriminator_cat_1/dropout/mul_grad/tuple/control_dependency_1Identity:gradients_2/discriminator_cat_1/dropout/mul_grad/Reshape_1B^gradients_2/discriminator_cat_1/dropout/mul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_2/discriminator_cat_1/dropout/mul_grad/Reshape_1
 
.gradients_2/discriminator_1/Relu_grad/ReluGradReluGradEgradients_2/discriminator_1/dropout/mul_grad/tuple/control_dependencydiscriminator_1/Relu*
T0
¬
2gradients_2/discriminator_cat_1/Relu_grad/ReluGradReluGradIgradients_2/discriminator_cat_1/dropout/mul_grad/tuple/control_dependencydiscriminator_cat_1/Relu*
T0
d
*gradients_2/discriminator_1/add_grad/ShapeShapediscriminator_1/MatMul*
T0*
out_type0
e
,gradients_2/discriminator_1/add_grad/Shape_1Shapediscriminator/b0/read*
T0*
out_type0
¶
:gradients_2/discriminator_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_2/discriminator_1/add_grad/Shape,gradients_2/discriminator_1/add_grad/Shape_1*
T0
Į
(gradients_2/discriminator_1/add_grad/SumSum.gradients_2/discriminator_1/Relu_grad/ReluGrad:gradients_2/discriminator_1/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
¤
,gradients_2/discriminator_1/add_grad/ReshapeReshape(gradients_2/discriminator_1/add_grad/Sum*gradients_2/discriminator_1/add_grad/Shape*
T0*
Tshape0
Å
*gradients_2/discriminator_1/add_grad/Sum_1Sum.gradients_2/discriminator_1/Relu_grad/ReluGrad<gradients_2/discriminator_1/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
Ŗ
.gradients_2/discriminator_1/add_grad/Reshape_1Reshape*gradients_2/discriminator_1/add_grad/Sum_1,gradients_2/discriminator_1/add_grad/Shape_1*
T0*
Tshape0

5gradients_2/discriminator_1/add_grad/tuple/group_depsNoOp-^gradients_2/discriminator_1/add_grad/Reshape/^gradients_2/discriminator_1/add_grad/Reshape_1
ł
=gradients_2/discriminator_1/add_grad/tuple/control_dependencyIdentity,gradients_2/discriminator_1/add_grad/Reshape6^gradients_2/discriminator_1/add_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_2/discriminator_1/add_grad/Reshape
’
?gradients_2/discriminator_1/add_grad/tuple/control_dependency_1Identity.gradients_2/discriminator_1/add_grad/Reshape_16^gradients_2/discriminator_1/add_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_2/discriminator_1/add_grad/Reshape_1
l
.gradients_2/discriminator_cat_1/add_grad/ShapeShapediscriminator_cat_1/MatMul*
T0*
out_type0
m
0gradients_2/discriminator_cat_1/add_grad/Shape_1Shapediscriminator_cat/b0/read*
T0*
out_type0
Ā
>gradients_2/discriminator_cat_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients_2/discriminator_cat_1/add_grad/Shape0gradients_2/discriminator_cat_1/add_grad/Shape_1*
T0
Ķ
,gradients_2/discriminator_cat_1/add_grad/SumSum2gradients_2/discriminator_cat_1/Relu_grad/ReluGrad>gradients_2/discriminator_cat_1/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
°
0gradients_2/discriminator_cat_1/add_grad/ReshapeReshape,gradients_2/discriminator_cat_1/add_grad/Sum.gradients_2/discriminator_cat_1/add_grad/Shape*
T0*
Tshape0
Ń
.gradients_2/discriminator_cat_1/add_grad/Sum_1Sum2gradients_2/discriminator_cat_1/Relu_grad/ReluGrad@gradients_2/discriminator_cat_1/add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
¶
2gradients_2/discriminator_cat_1/add_grad/Reshape_1Reshape.gradients_2/discriminator_cat_1/add_grad/Sum_10gradients_2/discriminator_cat_1/add_grad/Shape_1*
T0*
Tshape0
©
9gradients_2/discriminator_cat_1/add_grad/tuple/group_depsNoOp1^gradients_2/discriminator_cat_1/add_grad/Reshape3^gradients_2/discriminator_cat_1/add_grad/Reshape_1

Agradients_2/discriminator_cat_1/add_grad/tuple/control_dependencyIdentity0gradients_2/discriminator_cat_1/add_grad/Reshape:^gradients_2/discriminator_cat_1/add_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_2/discriminator_cat_1/add_grad/Reshape

Cgradients_2/discriminator_cat_1/add_grad/tuple/control_dependency_1Identity2gradients_2/discriminator_cat_1/add_grad/Reshape_1:^gradients_2/discriminator_cat_1/add_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_2/discriminator_cat_1/add_grad/Reshape_1
½
.gradients_2/discriminator_1/MatMul_grad/MatMulMatMul=gradients_2/discriminator_1/add_grad/tuple/control_dependencydiscriminator/w0/read*
transpose_b(*
T0*
transpose_a( 
Ę
0gradients_2/discriminator_1/MatMul_grad/MatMul_1MatMulCNN_encoder_cat/zout/BiasAdd=gradients_2/discriminator_1/add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
¤
8gradients_2/discriminator_1/MatMul_grad/tuple/group_depsNoOp/^gradients_2/discriminator_1/MatMul_grad/MatMul1^gradients_2/discriminator_1/MatMul_grad/MatMul_1

@gradients_2/discriminator_1/MatMul_grad/tuple/control_dependencyIdentity.gradients_2/discriminator_1/MatMul_grad/MatMul9^gradients_2/discriminator_1/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_2/discriminator_1/MatMul_grad/MatMul

Bgradients_2/discriminator_1/MatMul_grad/tuple/control_dependency_1Identity0gradients_2/discriminator_1/MatMul_grad/MatMul_19^gradients_2/discriminator_1/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_2/discriminator_1/MatMul_grad/MatMul_1
É
2gradients_2/discriminator_cat_1/MatMul_grad/MatMulMatMulAgradients_2/discriminator_cat_1/add_grad/tuple/control_dependencydiscriminator_cat/w0/read*
transpose_b(*
T0*
transpose_a( 
Š
4gradients_2/discriminator_cat_1/MatMul_grad/MatMul_1MatMulCNN_encoder_cat/catout/SoftmaxAgradients_2/discriminator_cat_1/add_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
°
<gradients_2/discriminator_cat_1/MatMul_grad/tuple/group_depsNoOp3^gradients_2/discriminator_cat_1/MatMul_grad/MatMul5^gradients_2/discriminator_cat_1/MatMul_grad/MatMul_1

Dgradients_2/discriminator_cat_1/MatMul_grad/tuple/control_dependencyIdentity2gradients_2/discriminator_cat_1/MatMul_grad/MatMul=^gradients_2/discriminator_cat_1/MatMul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_2/discriminator_cat_1/MatMul_grad/MatMul

Fgradients_2/discriminator_cat_1/MatMul_grad/tuple/control_dependency_1Identity4gradients_2/discriminator_cat_1/MatMul_grad/MatMul_1=^gradients_2/discriminator_cat_1/MatMul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_2/discriminator_cat_1/MatMul_grad/MatMul_1
Ŗ
9gradients_2/CNN_encoder_cat/zout/BiasAdd_grad/BiasAddGradBiasAddGrad@gradients_2/discriminator_1/MatMul_grad/tuple/control_dependency*
T0*
data_formatNHWC
Å
>gradients_2/CNN_encoder_cat/zout/BiasAdd_grad/tuple/group_depsNoOp:^gradients_2/CNN_encoder_cat/zout/BiasAdd_grad/BiasAddGradA^gradients_2/discriminator_1/MatMul_grad/tuple/control_dependency
”
Fgradients_2/CNN_encoder_cat/zout/BiasAdd_grad/tuple/control_dependencyIdentity@gradients_2/discriminator_1/MatMul_grad/tuple/control_dependency?^gradients_2/CNN_encoder_cat/zout/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_2/discriminator_1/MatMul_grad/MatMul
§
Hgradients_2/CNN_encoder_cat/zout/BiasAdd_grad/tuple/control_dependency_1Identity9gradients_2/CNN_encoder_cat/zout/BiasAdd_grad/BiasAddGrad?^gradients_2/CNN_encoder_cat/zout/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_2/CNN_encoder_cat/zout/BiasAdd_grad/BiasAddGrad
©
3gradients_2/CNN_encoder_cat/catout/Softmax_grad/mulMulDgradients_2/discriminator_cat_1/MatMul_grad/tuple/control_dependencyCNN_encoder_cat/catout/Softmax*
T0
x
Egradients_2/CNN_encoder_cat/catout/Softmax_grad/Sum/reduction_indicesConst*
valueB :
’’’’’’’’’*
dtype0
Ü
3gradients_2/CNN_encoder_cat/catout/Softmax_grad/SumSum3gradients_2/CNN_encoder_cat/catout/Softmax_grad/mulEgradients_2/CNN_encoder_cat/catout/Softmax_grad/Sum/reduction_indices*

Tidx0*
	keep_dims(*
T0
¾
3gradients_2/CNN_encoder_cat/catout/Softmax_grad/subSubDgradients_2/discriminator_cat_1/MatMul_grad/tuple/control_dependency3gradients_2/CNN_encoder_cat/catout/Softmax_grad/Sum*
T0

5gradients_2/CNN_encoder_cat/catout/Softmax_grad/mul_1Mul3gradients_2/CNN_encoder_cat/catout/Softmax_grad/subCNN_encoder_cat/catout/Softmax*
T0
Ń
3gradients_2/CNN_encoder_cat/zout/MatMul_grad/MatMulMatMulFgradients_2/CNN_encoder_cat/zout/BiasAdd_grad/tuple/control_dependencyCNN_encoder_cat/zout/W/read*
transpose_b(*
T0*
transpose_a( 
Ō
5gradients_2/CNN_encoder_cat/zout/MatMul_grad/MatMul_1MatMulCNN_encoder_cat/zout/ReshapeFgradients_2/CNN_encoder_cat/zout/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
³
=gradients_2/CNN_encoder_cat/zout/MatMul_grad/tuple/group_depsNoOp4^gradients_2/CNN_encoder_cat/zout/MatMul_grad/MatMul6^gradients_2/CNN_encoder_cat/zout/MatMul_grad/MatMul_1

Egradients_2/CNN_encoder_cat/zout/MatMul_grad/tuple/control_dependencyIdentity3gradients_2/CNN_encoder_cat/zout/MatMul_grad/MatMul>^gradients_2/CNN_encoder_cat/zout/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@gradients_2/CNN_encoder_cat/zout/MatMul_grad/MatMul

Ggradients_2/CNN_encoder_cat/zout/MatMul_grad/tuple/control_dependency_1Identity5gradients_2/CNN_encoder_cat/zout/MatMul_grad/MatMul_1>^gradients_2/CNN_encoder_cat/zout/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_2/CNN_encoder_cat/zout/MatMul_grad/MatMul_1
”
;gradients_2/CNN_encoder_cat/catout/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients_2/CNN_encoder_cat/catout/Softmax_grad/mul_1*
T0*
data_formatNHWC
¾
@gradients_2/CNN_encoder_cat/catout/BiasAdd_grad/tuple/group_depsNoOp<^gradients_2/CNN_encoder_cat/catout/BiasAdd_grad/BiasAddGrad6^gradients_2/CNN_encoder_cat/catout/Softmax_grad/mul_1
”
Hgradients_2/CNN_encoder_cat/catout/BiasAdd_grad/tuple/control_dependencyIdentity5gradients_2/CNN_encoder_cat/catout/Softmax_grad/mul_1A^gradients_2/CNN_encoder_cat/catout/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_2/CNN_encoder_cat/catout/Softmax_grad/mul_1
Æ
Jgradients_2/CNN_encoder_cat/catout/BiasAdd_grad/tuple/control_dependency_1Identity;gradients_2/CNN_encoder_cat/catout/BiasAdd_grad/BiasAddGradA^gradients_2/CNN_encoder_cat/catout/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_2/CNN_encoder_cat/catout/BiasAdd_grad/BiasAddGrad
z
3gradients_2/CNN_encoder_cat/zout/Reshape_grad/ShapeShape#CNN_encoder_cat/MaxPool2D_2/MaxPool*
T0*
out_type0
Ó
5gradients_2/CNN_encoder_cat/zout/Reshape_grad/ReshapeReshapeEgradients_2/CNN_encoder_cat/zout/MatMul_grad/tuple/control_dependency3gradients_2/CNN_encoder_cat/zout/Reshape_grad/Shape*
T0*
Tshape0
×
5gradients_2/CNN_encoder_cat/catout/MatMul_grad/MatMulMatMulHgradients_2/CNN_encoder_cat/catout/BiasAdd_grad/tuple/control_dependencyCNN_encoder_cat/catout/W/read*
transpose_b(*
T0*
transpose_a( 
Ś
7gradients_2/CNN_encoder_cat/catout/MatMul_grad/MatMul_1MatMulCNN_encoder_cat/catout/ReshapeHgradients_2/CNN_encoder_cat/catout/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
¹
?gradients_2/CNN_encoder_cat/catout/MatMul_grad/tuple/group_depsNoOp6^gradients_2/CNN_encoder_cat/catout/MatMul_grad/MatMul8^gradients_2/CNN_encoder_cat/catout/MatMul_grad/MatMul_1

Ggradients_2/CNN_encoder_cat/catout/MatMul_grad/tuple/control_dependencyIdentity5gradients_2/CNN_encoder_cat/catout/MatMul_grad/MatMul@^gradients_2/CNN_encoder_cat/catout/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_2/CNN_encoder_cat/catout/MatMul_grad/MatMul
„
Igradients_2/CNN_encoder_cat/catout/MatMul_grad/tuple/control_dependency_1Identity7gradients_2/CNN_encoder_cat/catout/MatMul_grad/MatMul_1@^gradients_2/CNN_encoder_cat/catout/MatMul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_2/CNN_encoder_cat/catout/MatMul_grad/MatMul_1
|
5gradients_2/CNN_encoder_cat/catout/Reshape_grad/ShapeShape#CNN_encoder_cat/MaxPool2D_2/MaxPool*
T0*
out_type0
Ł
7gradients_2/CNN_encoder_cat/catout/Reshape_grad/ReshapeReshapeGgradients_2/CNN_encoder_cat/catout/MatMul_grad/tuple/control_dependency5gradients_2/CNN_encoder_cat/catout/Reshape_grad/Shape*
T0*
Tshape0
ę
gradients_2/AddN_2AddN5gradients_2/CNN_encoder_cat/zout/Reshape_grad/Reshape7gradients_2/CNN_encoder_cat/catout/Reshape_grad/Reshape*
T0*H
_class>
<:loc:@gradients_2/CNN_encoder_cat/zout/Reshape_grad/Reshape*
N

@gradients_2/CNN_encoder_cat/MaxPool2D_2/MaxPool_grad/MaxPoolGradMaxPoolGradCNN_encoder_cat/Conv2D_2/Tanh#CNN_encoder_cat/MaxPool2D_2/MaxPoolgradients_2/AddN_2*
ksize
*
paddingSAME*
T0*
data_formatNHWC*
strides

­
7gradients_2/CNN_encoder_cat/Conv2D_2/Tanh_grad/TanhGradTanhGradCNN_encoder_cat/Conv2D_2/Tanh@gradients_2/CNN_encoder_cat/MaxPool2D_2/MaxPool_grad/MaxPoolGrad*
T0
„
=gradients_2/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients_2/CNN_encoder_cat/Conv2D_2/Tanh_grad/TanhGrad*
T0*
data_formatNHWC
Ä
Bgradients_2/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/group_depsNoOp>^gradients_2/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/BiasAddGrad8^gradients_2/CNN_encoder_cat/Conv2D_2/Tanh_grad/TanhGrad
©
Jgradients_2/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/control_dependencyIdentity7gradients_2/CNN_encoder_cat/Conv2D_2/Tanh_grad/TanhGradC^gradients_2/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_2/CNN_encoder_cat/Conv2D_2/Tanh_grad/TanhGrad
·
Lgradients_2/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/control_dependency_1Identity=gradients_2/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/BiasAddGradC^gradients_2/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_2/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/BiasAddGrad
©
7gradients_2/CNN_encoder_cat/Conv2D_2/Conv2D_grad/ShapeNShapeN#CNN_encoder_cat/MaxPool2D_1/MaxPoolCNN_encoder_cat/Conv2D_2/W/read*
T0*
out_type0*
N

Dgradients_2/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput7gradients_2/CNN_encoder_cat/Conv2D_2/Conv2D_grad/ShapeNCNN_encoder_cat/Conv2D_2/W/readJgradients_2/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME

Egradients_2/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter#CNN_encoder_cat/MaxPool2D_1/MaxPool9gradients_2/CNN_encoder_cat/Conv2D_2/Conv2D_grad/ShapeN:1Jgradients_2/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
Ų
Agradients_2/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/group_depsNoOpF^gradients_2/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropFilterE^gradients_2/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropInput
Į
Igradients_2/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/control_dependencyIdentityDgradients_2/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropInputB^gradients_2/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_2/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropInput
Å
Kgradients_2/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/control_dependency_1IdentityEgradients_2/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropFilterB^gradients_2/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_2/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropFilter
¹
@gradients_2/CNN_encoder_cat/MaxPool2D_1/MaxPool_grad/MaxPoolGradMaxPoolGradCNN_encoder_cat/Conv2D_1/Tanh#CNN_encoder_cat/MaxPool2D_1/MaxPoolIgradients_2/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*
T0*
data_formatNHWC*
strides

­
7gradients_2/CNN_encoder_cat/Conv2D_1/Tanh_grad/TanhGradTanhGradCNN_encoder_cat/Conv2D_1/Tanh@gradients_2/CNN_encoder_cat/MaxPool2D_1/MaxPool_grad/MaxPoolGrad*
T0
„
=gradients_2/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients_2/CNN_encoder_cat/Conv2D_1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC
Ä
Bgradients_2/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/group_depsNoOp>^gradients_2/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/BiasAddGrad8^gradients_2/CNN_encoder_cat/Conv2D_1/Tanh_grad/TanhGrad
©
Jgradients_2/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/control_dependencyIdentity7gradients_2/CNN_encoder_cat/Conv2D_1/Tanh_grad/TanhGradC^gradients_2/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_2/CNN_encoder_cat/Conv2D_1/Tanh_grad/TanhGrad
·
Lgradients_2/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/control_dependency_1Identity=gradients_2/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/BiasAddGradC^gradients_2/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_2/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/BiasAddGrad
§
7gradients_2/CNN_encoder_cat/Conv2D_1/Conv2D_grad/ShapeNShapeN!CNN_encoder_cat/MaxPool2D/MaxPoolCNN_encoder_cat/Conv2D_1/W/read*
T0*
out_type0*
N

Dgradients_2/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput7gradients_2/CNN_encoder_cat/Conv2D_1/Conv2D_grad/ShapeNCNN_encoder_cat/Conv2D_1/W/readJgradients_2/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME

Egradients_2/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter!CNN_encoder_cat/MaxPool2D/MaxPool9gradients_2/CNN_encoder_cat/Conv2D_1/Conv2D_grad/ShapeN:1Jgradients_2/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
Ų
Agradients_2/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/group_depsNoOpF^gradients_2/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropFilterE^gradients_2/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropInput
Į
Igradients_2/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/control_dependencyIdentityDgradients_2/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropInputB^gradients_2/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_2/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropInput
Å
Kgradients_2/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/control_dependency_1IdentityEgradients_2/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropFilterB^gradients_2/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_2/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropFilter
³
>gradients_2/CNN_encoder_cat/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradCNN_encoder_cat/Conv2D/Tanh!CNN_encoder_cat/MaxPool2D/MaxPoolIgradients_2/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*
T0*
data_formatNHWC*
strides

§
5gradients_2/CNN_encoder_cat/Conv2D/Tanh_grad/TanhGradTanhGradCNN_encoder_cat/Conv2D/Tanh>gradients_2/CNN_encoder_cat/MaxPool2D/MaxPool_grad/MaxPoolGrad*
T0
”
;gradients_2/CNN_encoder_cat/Conv2D/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients_2/CNN_encoder_cat/Conv2D/Tanh_grad/TanhGrad*
T0*
data_formatNHWC
¾
@gradients_2/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/group_depsNoOp<^gradients_2/CNN_encoder_cat/Conv2D/BiasAdd_grad/BiasAddGrad6^gradients_2/CNN_encoder_cat/Conv2D/Tanh_grad/TanhGrad
”
Hgradients_2/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/control_dependencyIdentity5gradients_2/CNN_encoder_cat/Conv2D/Tanh_grad/TanhGradA^gradients_2/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_2/CNN_encoder_cat/Conv2D/Tanh_grad/TanhGrad
Æ
Jgradients_2/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/control_dependency_1Identity;gradients_2/CNN_encoder_cat/Conv2D/BiasAdd_grad/BiasAddGradA^gradients_2/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_2/CNN_encoder_cat/Conv2D/BiasAdd_grad/BiasAddGrad

5gradients_2/CNN_encoder_cat/Conv2D/Conv2D_grad/ShapeNShapeNCNN_encoder_cat/ReshapeCNN_encoder_cat/Conv2D/W/read*
T0*
out_type0*
N

Bgradients_2/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput5gradients_2/CNN_encoder_cat/Conv2D/Conv2D_grad/ShapeNCNN_encoder_cat/Conv2D/W/readHgradients_2/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME

Cgradients_2/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterCNN_encoder_cat/Reshape7gradients_2/CNN_encoder_cat/Conv2D/Conv2D_grad/ShapeN:1Hgradients_2/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
Ņ
?gradients_2/CNN_encoder_cat/Conv2D/Conv2D_grad/tuple/group_depsNoOpD^gradients_2/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropFilterC^gradients_2/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropInput
¹
Ggradients_2/CNN_encoder_cat/Conv2D/Conv2D_grad/tuple/control_dependencyIdentityBgradients_2/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropInput@^gradients_2/CNN_encoder_cat/Conv2D/Conv2D_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_2/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropInput
½
Igradients_2/CNN_encoder_cat/Conv2D/Conv2D_grad/tuple/control_dependency_1IdentityCgradients_2/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropFilter@^gradients_2/CNN_encoder_cat/Conv2D/Conv2D_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_2/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropFilter
u
beta1_power_2/initial_valueConst*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
valueB
 *fff?*
dtype0

beta1_power_2
VariableV2*
shape: *
shared_name *+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
dtype0*
	container 
©
beta1_power_2/AssignAssignbeta1_power_2beta1_power_2/initial_value*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(
c
beta1_power_2/readIdentitybeta1_power_2*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W
u
beta2_power_2/initial_valueConst*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
valueB
 *w¾?*
dtype0

beta2_power_2
VariableV2*
shape: *
shared_name *+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
dtype0*
	container 
©
beta2_power_2/AssignAssignbeta2_power_2beta2_power_2/initial_value*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(
c
beta2_power_2/readIdentitybeta2_power_2*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W

1CNN_encoder_cat/Conv2D/W/Adam_4/Initializer/zerosConst*%
valueB *    *+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
dtype0
Ø
CNN_encoder_cat/Conv2D/W/Adam_4
VariableV2*
shape: *
shared_name *+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
dtype0*
	container 
ć
&CNN_encoder_cat/Conv2D/W/Adam_4/AssignAssignCNN_encoder_cat/Conv2D/W/Adam_41CNN_encoder_cat/Conv2D/W/Adam_4/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(

$CNN_encoder_cat/Conv2D/W/Adam_4/readIdentityCNN_encoder_cat/Conv2D/W/Adam_4*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W

1CNN_encoder_cat/Conv2D/W/Adam_5/Initializer/zerosConst*%
valueB *    *+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
dtype0
Ø
CNN_encoder_cat/Conv2D/W/Adam_5
VariableV2*
shape: *
shared_name *+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
dtype0*
	container 
ć
&CNN_encoder_cat/Conv2D/W/Adam_5/AssignAssignCNN_encoder_cat/Conv2D/W/Adam_51CNN_encoder_cat/Conv2D/W/Adam_5/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(

$CNN_encoder_cat/Conv2D/W/Adam_5/readIdentityCNN_encoder_cat/Conv2D/W/Adam_5*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W

1CNN_encoder_cat/Conv2D/b/Adam_4/Initializer/zerosConst*
valueB *    *+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
dtype0

CNN_encoder_cat/Conv2D/b/Adam_4
VariableV2*
shape: *
shared_name *+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
dtype0*
	container 
ć
&CNN_encoder_cat/Conv2D/b/Adam_4/AssignAssignCNN_encoder_cat/Conv2D/b/Adam_41CNN_encoder_cat/Conv2D/b/Adam_4/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
validate_shape(

$CNN_encoder_cat/Conv2D/b/Adam_4/readIdentityCNN_encoder_cat/Conv2D/b/Adam_4*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b

1CNN_encoder_cat/Conv2D/b/Adam_5/Initializer/zerosConst*
valueB *    *+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
dtype0

CNN_encoder_cat/Conv2D/b/Adam_5
VariableV2*
shape: *
shared_name *+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
dtype0*
	container 
ć
&CNN_encoder_cat/Conv2D/b/Adam_5/AssignAssignCNN_encoder_cat/Conv2D/b/Adam_51CNN_encoder_cat/Conv2D/b/Adam_5/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
validate_shape(

$CNN_encoder_cat/Conv2D/b/Adam_5/readIdentityCNN_encoder_cat/Conv2D/b/Adam_5*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b
Æ
CCNN_encoder_cat/Conv2D_1/W/Adam_4/Initializer/zeros/shape_as_tensorConst*%
valueB"              *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0

9CNN_encoder_cat/Conv2D_1/W/Adam_4/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0

3CNN_encoder_cat/Conv2D_1/W/Adam_4/Initializer/zerosFillCCNN_encoder_cat/Conv2D_1/W/Adam_4/Initializer/zeros/shape_as_tensor9CNN_encoder_cat/Conv2D_1/W/Adam_4/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W
¬
!CNN_encoder_cat/Conv2D_1/W/Adam_4
VariableV2*
shape:  *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_1/W/Adam_4/AssignAssign!CNN_encoder_cat/Conv2D_1/W/Adam_43CNN_encoder_cat/Conv2D_1/W/Adam_4/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
validate_shape(

&CNN_encoder_cat/Conv2D_1/W/Adam_4/readIdentity!CNN_encoder_cat/Conv2D_1/W/Adam_4*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W
Æ
CCNN_encoder_cat/Conv2D_1/W/Adam_5/Initializer/zeros/shape_as_tensorConst*%
valueB"              *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0

9CNN_encoder_cat/Conv2D_1/W/Adam_5/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0

3CNN_encoder_cat/Conv2D_1/W/Adam_5/Initializer/zerosFillCCNN_encoder_cat/Conv2D_1/W/Adam_5/Initializer/zeros/shape_as_tensor9CNN_encoder_cat/Conv2D_1/W/Adam_5/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W
¬
!CNN_encoder_cat/Conv2D_1/W/Adam_5
VariableV2*
shape:  *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_1/W/Adam_5/AssignAssign!CNN_encoder_cat/Conv2D_1/W/Adam_53CNN_encoder_cat/Conv2D_1/W/Adam_5/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
validate_shape(

&CNN_encoder_cat/Conv2D_1/W/Adam_5/readIdentity!CNN_encoder_cat/Conv2D_1/W/Adam_5*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W

3CNN_encoder_cat/Conv2D_1/b/Adam_4/Initializer/zerosConst*
valueB *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
dtype0
 
!CNN_encoder_cat/Conv2D_1/b/Adam_4
VariableV2*
shape: *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_1/b/Adam_4/AssignAssign!CNN_encoder_cat/Conv2D_1/b/Adam_43CNN_encoder_cat/Conv2D_1/b/Adam_4/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
validate_shape(

&CNN_encoder_cat/Conv2D_1/b/Adam_4/readIdentity!CNN_encoder_cat/Conv2D_1/b/Adam_4*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b

3CNN_encoder_cat/Conv2D_1/b/Adam_5/Initializer/zerosConst*
valueB *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
dtype0
 
!CNN_encoder_cat/Conv2D_1/b/Adam_5
VariableV2*
shape: *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_1/b/Adam_5/AssignAssign!CNN_encoder_cat/Conv2D_1/b/Adam_53CNN_encoder_cat/Conv2D_1/b/Adam_5/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
validate_shape(

&CNN_encoder_cat/Conv2D_1/b/Adam_5/readIdentity!CNN_encoder_cat/Conv2D_1/b/Adam_5*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b
Æ
CCNN_encoder_cat/Conv2D_2/W/Adam_4/Initializer/zeros/shape_as_tensorConst*%
valueB"              *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0

9CNN_encoder_cat/Conv2D_2/W/Adam_4/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0

3CNN_encoder_cat/Conv2D_2/W/Adam_4/Initializer/zerosFillCCNN_encoder_cat/Conv2D_2/W/Adam_4/Initializer/zeros/shape_as_tensor9CNN_encoder_cat/Conv2D_2/W/Adam_4/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W
¬
!CNN_encoder_cat/Conv2D_2/W/Adam_4
VariableV2*
shape:  *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_2/W/Adam_4/AssignAssign!CNN_encoder_cat/Conv2D_2/W/Adam_43CNN_encoder_cat/Conv2D_2/W/Adam_4/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
validate_shape(

&CNN_encoder_cat/Conv2D_2/W/Adam_4/readIdentity!CNN_encoder_cat/Conv2D_2/W/Adam_4*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W
Æ
CCNN_encoder_cat/Conv2D_2/W/Adam_5/Initializer/zeros/shape_as_tensorConst*%
valueB"              *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0

9CNN_encoder_cat/Conv2D_2/W/Adam_5/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0

3CNN_encoder_cat/Conv2D_2/W/Adam_5/Initializer/zerosFillCCNN_encoder_cat/Conv2D_2/W/Adam_5/Initializer/zeros/shape_as_tensor9CNN_encoder_cat/Conv2D_2/W/Adam_5/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W
¬
!CNN_encoder_cat/Conv2D_2/W/Adam_5
VariableV2*
shape:  *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_2/W/Adam_5/AssignAssign!CNN_encoder_cat/Conv2D_2/W/Adam_53CNN_encoder_cat/Conv2D_2/W/Adam_5/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
validate_shape(

&CNN_encoder_cat/Conv2D_2/W/Adam_5/readIdentity!CNN_encoder_cat/Conv2D_2/W/Adam_5*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W

3CNN_encoder_cat/Conv2D_2/b/Adam_4/Initializer/zerosConst*
valueB *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
dtype0
 
!CNN_encoder_cat/Conv2D_2/b/Adam_4
VariableV2*
shape: *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_2/b/Adam_4/AssignAssign!CNN_encoder_cat/Conv2D_2/b/Adam_43CNN_encoder_cat/Conv2D_2/b/Adam_4/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
validate_shape(

&CNN_encoder_cat/Conv2D_2/b/Adam_4/readIdentity!CNN_encoder_cat/Conv2D_2/b/Adam_4*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b

3CNN_encoder_cat/Conv2D_2/b/Adam_5/Initializer/zerosConst*
valueB *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
dtype0
 
!CNN_encoder_cat/Conv2D_2/b/Adam_5
VariableV2*
shape: *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_2/b/Adam_5/AssignAssign!CNN_encoder_cat/Conv2D_2/b/Adam_53CNN_encoder_cat/Conv2D_2/b/Adam_5/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
validate_shape(

&CNN_encoder_cat/Conv2D_2/b/Adam_5/readIdentity!CNN_encoder_cat/Conv2D_2/b/Adam_5*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b
£
ACNN_encoder_cat/catout/W/Adam_4/Initializer/zeros/shape_as_tensorConst*
valueB"     *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0

7CNN_encoder_cat/catout/W/Adam_4/Initializer/zeros/ConstConst*
valueB
 *    *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0
ż
1CNN_encoder_cat/catout/W/Adam_4/Initializer/zerosFillACNN_encoder_cat/catout/W/Adam_4/Initializer/zeros/shape_as_tensor7CNN_encoder_cat/catout/W/Adam_4/Initializer/zeros/Const*
T0*

index_type0*+
_class!
loc:@CNN_encoder_cat/catout/W
”
CNN_encoder_cat/catout/W/Adam_4
VariableV2*
shape:	*
shared_name *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0*
	container 
ć
&CNN_encoder_cat/catout/W/Adam_4/AssignAssignCNN_encoder_cat/catout/W/Adam_41CNN_encoder_cat/catout/W/Adam_4/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W*
validate_shape(

$CNN_encoder_cat/catout/W/Adam_4/readIdentityCNN_encoder_cat/catout/W/Adam_4*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W
£
ACNN_encoder_cat/catout/W/Adam_5/Initializer/zeros/shape_as_tensorConst*
valueB"     *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0

7CNN_encoder_cat/catout/W/Adam_5/Initializer/zeros/ConstConst*
valueB
 *    *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0
ż
1CNN_encoder_cat/catout/W/Adam_5/Initializer/zerosFillACNN_encoder_cat/catout/W/Adam_5/Initializer/zeros/shape_as_tensor7CNN_encoder_cat/catout/W/Adam_5/Initializer/zeros/Const*
T0*

index_type0*+
_class!
loc:@CNN_encoder_cat/catout/W
”
CNN_encoder_cat/catout/W/Adam_5
VariableV2*
shape:	*
shared_name *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0*
	container 
ć
&CNN_encoder_cat/catout/W/Adam_5/AssignAssignCNN_encoder_cat/catout/W/Adam_51CNN_encoder_cat/catout/W/Adam_5/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W*
validate_shape(

$CNN_encoder_cat/catout/W/Adam_5/readIdentityCNN_encoder_cat/catout/W/Adam_5*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W

1CNN_encoder_cat/catout/b/Adam_4/Initializer/zerosConst*
valueB*    *+
_class!
loc:@CNN_encoder_cat/catout/b*
dtype0

CNN_encoder_cat/catout/b/Adam_4
VariableV2*
shape:*
shared_name *+
_class!
loc:@CNN_encoder_cat/catout/b*
dtype0*
	container 
ć
&CNN_encoder_cat/catout/b/Adam_4/AssignAssignCNN_encoder_cat/catout/b/Adam_41CNN_encoder_cat/catout/b/Adam_4/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b*
validate_shape(

$CNN_encoder_cat/catout/b/Adam_4/readIdentityCNN_encoder_cat/catout/b/Adam_4*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b

1CNN_encoder_cat/catout/b/Adam_5/Initializer/zerosConst*
valueB*    *+
_class!
loc:@CNN_encoder_cat/catout/b*
dtype0

CNN_encoder_cat/catout/b/Adam_5
VariableV2*
shape:*
shared_name *+
_class!
loc:@CNN_encoder_cat/catout/b*
dtype0*
	container 
ć
&CNN_encoder_cat/catout/b/Adam_5/AssignAssignCNN_encoder_cat/catout/b/Adam_51CNN_encoder_cat/catout/b/Adam_5/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b*
validate_shape(

$CNN_encoder_cat/catout/b/Adam_5/readIdentityCNN_encoder_cat/catout/b/Adam_5*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b

?CNN_encoder_cat/zout/W/Adam_4/Initializer/zeros/shape_as_tensorConst*
valueB"  2   *)
_class
loc:@CNN_encoder_cat/zout/W*
dtype0

5CNN_encoder_cat/zout/W/Adam_4/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@CNN_encoder_cat/zout/W*
dtype0
õ
/CNN_encoder_cat/zout/W/Adam_4/Initializer/zerosFill?CNN_encoder_cat/zout/W/Adam_4/Initializer/zeros/shape_as_tensor5CNN_encoder_cat/zout/W/Adam_4/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@CNN_encoder_cat/zout/W

CNN_encoder_cat/zout/W/Adam_4
VariableV2*
shape:	2*
shared_name *)
_class
loc:@CNN_encoder_cat/zout/W*
dtype0*
	container 
Ū
$CNN_encoder_cat/zout/W/Adam_4/AssignAssignCNN_encoder_cat/zout/W/Adam_4/CNN_encoder_cat/zout/W/Adam_4/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/W*
validate_shape(

"CNN_encoder_cat/zout/W/Adam_4/readIdentityCNN_encoder_cat/zout/W/Adam_4*
T0*)
_class
loc:@CNN_encoder_cat/zout/W

?CNN_encoder_cat/zout/W/Adam_5/Initializer/zeros/shape_as_tensorConst*
valueB"  2   *)
_class
loc:@CNN_encoder_cat/zout/W*
dtype0

5CNN_encoder_cat/zout/W/Adam_5/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@CNN_encoder_cat/zout/W*
dtype0
õ
/CNN_encoder_cat/zout/W/Adam_5/Initializer/zerosFill?CNN_encoder_cat/zout/W/Adam_5/Initializer/zeros/shape_as_tensor5CNN_encoder_cat/zout/W/Adam_5/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@CNN_encoder_cat/zout/W

CNN_encoder_cat/zout/W/Adam_5
VariableV2*
shape:	2*
shared_name *)
_class
loc:@CNN_encoder_cat/zout/W*
dtype0*
	container 
Ū
$CNN_encoder_cat/zout/W/Adam_5/AssignAssignCNN_encoder_cat/zout/W/Adam_5/CNN_encoder_cat/zout/W/Adam_5/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/W*
validate_shape(

"CNN_encoder_cat/zout/W/Adam_5/readIdentityCNN_encoder_cat/zout/W/Adam_5*
T0*)
_class
loc:@CNN_encoder_cat/zout/W

/CNN_encoder_cat/zout/b/Adam_4/Initializer/zerosConst*
valueB2*    *)
_class
loc:@CNN_encoder_cat/zout/b*
dtype0

CNN_encoder_cat/zout/b/Adam_4
VariableV2*
shape:2*
shared_name *)
_class
loc:@CNN_encoder_cat/zout/b*
dtype0*
	container 
Ū
$CNN_encoder_cat/zout/b/Adam_4/AssignAssignCNN_encoder_cat/zout/b/Adam_4/CNN_encoder_cat/zout/b/Adam_4/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/b*
validate_shape(

"CNN_encoder_cat/zout/b/Adam_4/readIdentityCNN_encoder_cat/zout/b/Adam_4*
T0*)
_class
loc:@CNN_encoder_cat/zout/b

/CNN_encoder_cat/zout/b/Adam_5/Initializer/zerosConst*
valueB2*    *)
_class
loc:@CNN_encoder_cat/zout/b*
dtype0

CNN_encoder_cat/zout/b/Adam_5
VariableV2*
shape:2*
shared_name *)
_class
loc:@CNN_encoder_cat/zout/b*
dtype0*
	container 
Ū
$CNN_encoder_cat/zout/b/Adam_5/AssignAssignCNN_encoder_cat/zout/b/Adam_5/CNN_encoder_cat/zout/b/Adam_5/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/b*
validate_shape(

"CNN_encoder_cat/zout/b/Adam_5/readIdentityCNN_encoder_cat/zout/b/Adam_5*
T0*)
_class
loc:@CNN_encoder_cat/zout/b
A
Adam_2/learning_rateConst*
valueB
 *·Q8*
dtype0
9
Adam_2/beta1Const*
valueB
 *fff?*
dtype0
9
Adam_2/beta2Const*
valueB
 *w¾?*
dtype0
;
Adam_2/epsilonConst*
valueB
 *wĢ+2*
dtype0
«
0Adam_2/update_CNN_encoder_cat/Conv2D/W/ApplyAdam	ApplyAdamCNN_encoder_cat/Conv2D/WCNN_encoder_cat/Conv2D/W/Adam_4CNN_encoder_cat/Conv2D/W/Adam_5beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilonIgradients_2/CNN_encoder_cat/Conv2D/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
use_nesterov( 
¬
0Adam_2/update_CNN_encoder_cat/Conv2D/b/ApplyAdam	ApplyAdamCNN_encoder_cat/Conv2D/bCNN_encoder_cat/Conv2D/b/Adam_4CNN_encoder_cat/Conv2D/b/Adam_5beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilonJgradients_2/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
use_nesterov( 
·
2Adam_2/update_CNN_encoder_cat/Conv2D_1/W/ApplyAdam	ApplyAdamCNN_encoder_cat/Conv2D_1/W!CNN_encoder_cat/Conv2D_1/W/Adam_4!CNN_encoder_cat/Conv2D_1/W/Adam_5beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilonKgradients_2/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
use_nesterov( 
ø
2Adam_2/update_CNN_encoder_cat/Conv2D_1/b/ApplyAdam	ApplyAdamCNN_encoder_cat/Conv2D_1/b!CNN_encoder_cat/Conv2D_1/b/Adam_4!CNN_encoder_cat/Conv2D_1/b/Adam_5beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilonLgradients_2/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
use_nesterov( 
·
2Adam_2/update_CNN_encoder_cat/Conv2D_2/W/ApplyAdam	ApplyAdamCNN_encoder_cat/Conv2D_2/W!CNN_encoder_cat/Conv2D_2/W/Adam_4!CNN_encoder_cat/Conv2D_2/W/Adam_5beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilonKgradients_2/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
use_nesterov( 
ø
2Adam_2/update_CNN_encoder_cat/Conv2D_2/b/ApplyAdam	ApplyAdamCNN_encoder_cat/Conv2D_2/b!CNN_encoder_cat/Conv2D_2/b/Adam_4!CNN_encoder_cat/Conv2D_2/b/Adam_5beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilonLgradients_2/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
use_nesterov( 
«
0Adam_2/update_CNN_encoder_cat/catout/W/ApplyAdam	ApplyAdamCNN_encoder_cat/catout/WCNN_encoder_cat/catout/W/Adam_4CNN_encoder_cat/catout/W/Adam_5beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilonIgradients_2/CNN_encoder_cat/catout/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*+
_class!
loc:@CNN_encoder_cat/catout/W*
use_nesterov( 
¬
0Adam_2/update_CNN_encoder_cat/catout/b/ApplyAdam	ApplyAdamCNN_encoder_cat/catout/bCNN_encoder_cat/catout/b/Adam_4CNN_encoder_cat/catout/b/Adam_5beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilonJgradients_2/CNN_encoder_cat/catout/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*+
_class!
loc:@CNN_encoder_cat/catout/b*
use_nesterov( 

.Adam_2/update_CNN_encoder_cat/zout/W/ApplyAdam	ApplyAdamCNN_encoder_cat/zout/WCNN_encoder_cat/zout/W/Adam_4CNN_encoder_cat/zout/W/Adam_5beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilonGgradients_2/CNN_encoder_cat/zout/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@CNN_encoder_cat/zout/W*
use_nesterov( 
 
.Adam_2/update_CNN_encoder_cat/zout/b/ApplyAdam	ApplyAdamCNN_encoder_cat/zout/bCNN_encoder_cat/zout/b/Adam_4CNN_encoder_cat/zout/b/Adam_5beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilonHgradients_2/CNN_encoder_cat/zout/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@CNN_encoder_cat/zout/b*
use_nesterov( 
ė

Adam_2/mulMulbeta1_power_2/readAdam_2/beta11^Adam_2/update_CNN_encoder_cat/Conv2D/W/ApplyAdam1^Adam_2/update_CNN_encoder_cat/Conv2D/b/ApplyAdam3^Adam_2/update_CNN_encoder_cat/Conv2D_1/W/ApplyAdam3^Adam_2/update_CNN_encoder_cat/Conv2D_1/b/ApplyAdam3^Adam_2/update_CNN_encoder_cat/Conv2D_2/W/ApplyAdam3^Adam_2/update_CNN_encoder_cat/Conv2D_2/b/ApplyAdam1^Adam_2/update_CNN_encoder_cat/catout/W/ApplyAdam1^Adam_2/update_CNN_encoder_cat/catout/b/ApplyAdam/^Adam_2/update_CNN_encoder_cat/zout/W/ApplyAdam/^Adam_2/update_CNN_encoder_cat/zout/b/ApplyAdam*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W

Adam_2/AssignAssignbeta1_power_2
Adam_2/mul*
use_locking( *
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(
ķ
Adam_2/mul_1Mulbeta2_power_2/readAdam_2/beta21^Adam_2/update_CNN_encoder_cat/Conv2D/W/ApplyAdam1^Adam_2/update_CNN_encoder_cat/Conv2D/b/ApplyAdam3^Adam_2/update_CNN_encoder_cat/Conv2D_1/W/ApplyAdam3^Adam_2/update_CNN_encoder_cat/Conv2D_1/b/ApplyAdam3^Adam_2/update_CNN_encoder_cat/Conv2D_2/W/ApplyAdam3^Adam_2/update_CNN_encoder_cat/Conv2D_2/b/ApplyAdam1^Adam_2/update_CNN_encoder_cat/catout/W/ApplyAdam1^Adam_2/update_CNN_encoder_cat/catout/b/ApplyAdam/^Adam_2/update_CNN_encoder_cat/zout/W/ApplyAdam/^Adam_2/update_CNN_encoder_cat/zout/b/ApplyAdam*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W

Adam_2/Assign_1Assignbeta2_power_2Adam_2/mul_1*
use_locking( *
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(
²
Adam_2NoOp^Adam_2/Assign^Adam_2/Assign_11^Adam_2/update_CNN_encoder_cat/Conv2D/W/ApplyAdam1^Adam_2/update_CNN_encoder_cat/Conv2D/b/ApplyAdam3^Adam_2/update_CNN_encoder_cat/Conv2D_1/W/ApplyAdam3^Adam_2/update_CNN_encoder_cat/Conv2D_1/b/ApplyAdam3^Adam_2/update_CNN_encoder_cat/Conv2D_2/W/ApplyAdam3^Adam_2/update_CNN_encoder_cat/Conv2D_2/b/ApplyAdam1^Adam_2/update_CNN_encoder_cat/catout/W/ApplyAdam1^Adam_2/update_CNN_encoder_cat/catout/b/ApplyAdam/^Adam_2/update_CNN_encoder_cat/zout/W/ApplyAdam/^Adam_2/update_CNN_encoder_cat/zout/b/ApplyAdam
:
gradients_3/ShapeConst*
valueB *
dtype0
B
gradients_3/grad_ys_0Const*
valueB
 *  ?*
dtype0
]
gradients_3/FillFillgradients_3/Shapegradients_3/grad_ys_0*
T0*

index_type0
S
%gradients_3/Mean_9_grad/Reshape/shapeConst*
valueB:*
dtype0
z
gradients_3/Mean_9_grad/ReshapeReshapegradients_3/Fill%gradients_3/Mean_9_grad/Reshape/shape*
T0*
Tshape0
o
gradients_3/Mean_9_grad/ShapeShape.softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0

gradients_3/Mean_9_grad/TileTilegradients_3/Mean_9_grad/Reshapegradients_3/Mean_9_grad/Shape*

Tmultiples0*
T0
q
gradients_3/Mean_9_grad/Shape_1Shape.softmax_cross_entropy_with_logits_sg/Reshape_2*
T0*
out_type0
H
gradients_3/Mean_9_grad/Shape_2Const*
valueB *
dtype0
K
gradients_3/Mean_9_grad/ConstConst*
valueB: *
dtype0

gradients_3/Mean_9_grad/ProdProdgradients_3/Mean_9_grad/Shape_1gradients_3/Mean_9_grad/Const*

Tidx0*
	keep_dims( *
T0
M
gradients_3/Mean_9_grad/Const_1Const*
valueB: *
dtype0

gradients_3/Mean_9_grad/Prod_1Prodgradients_3/Mean_9_grad/Shape_2gradients_3/Mean_9_grad/Const_1*

Tidx0*
	keep_dims( *
T0
K
!gradients_3/Mean_9_grad/Maximum/yConst*
value	B :*
dtype0
v
gradients_3/Mean_9_grad/MaximumMaximumgradients_3/Mean_9_grad/Prod_1!gradients_3/Mean_9_grad/Maximum/y*
T0
t
 gradients_3/Mean_9_grad/floordivFloorDivgradients_3/Mean_9_grad/Prodgradients_3/Mean_9_grad/Maximum*
T0
n
gradients_3/Mean_9_grad/CastCast gradients_3/Mean_9_grad/floordiv*

SrcT0*
Truncate( *

DstT0
o
gradients_3/Mean_9_grad/truedivRealDivgradients_3/Mean_9_grad/Tilegradients_3/Mean_9_grad/Cast*
T0

Egradients_3/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ShapeShape$softmax_cross_entropy_with_logits_sg*
T0*
out_type0
Ń
Ggradients_3/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeReshapegradients_3/Mean_9_grad/truedivEgradients_3/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/Shape*
T0*
Tshape0
T
gradients_3/zeros_like	ZerosLike&softmax_cross_entropy_with_logits_sg:1*
T0
w
Dgradients_3/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dimConst*
valueB :
’’’’’’’’’*
dtype0
ņ
@gradients_3/softmax_cross_entropy_with_logits_sg_grad/ExpandDims
ExpandDimsGgradients_3/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeDgradients_3/softmax_cross_entropy_with_logits_sg_grad/ExpandDims/dim*

Tdim0*
T0
³
9gradients_3/softmax_cross_entropy_with_logits_sg_grad/mulMul@gradients_3/softmax_cross_entropy_with_logits_sg_grad/ExpandDims&softmax_cross_entropy_with_logits_sg:1*
T0

@gradients_3/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax
LogSoftmax,softmax_cross_entropy_with_logits_sg/Reshape*
T0

9gradients_3/softmax_cross_entropy_with_logits_sg_grad/NegNeg@gradients_3/softmax_cross_entropy_with_logits_sg_grad/LogSoftmax*
T0
y
Fgradients_3/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dimConst*
valueB :
’’’’’’’’’*
dtype0
ö
Bgradients_3/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1
ExpandDimsGgradients_3/softmax_cross_entropy_with_logits_sg/Reshape_2_grad/ReshapeFgradients_3/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_1/dim*

Tdim0*
T0
Ź
;gradients_3/softmax_cross_entropy_with_logits_sg_grad/mul_1MulBgradients_3/softmax_cross_entropy_with_logits_sg_grad/ExpandDims_19gradients_3/softmax_cross_entropy_with_logits_sg_grad/Neg*
T0
Č
Fgradients_3/softmax_cross_entropy_with_logits_sg_grad/tuple/group_depsNoOp:^gradients_3/softmax_cross_entropy_with_logits_sg_grad/mul<^gradients_3/softmax_cross_entropy_with_logits_sg_grad/mul_1
µ
Ngradients_3/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyIdentity9gradients_3/softmax_cross_entropy_with_logits_sg_grad/mulG^gradients_3/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_3/softmax_cross_entropy_with_logits_sg_grad/mul
»
Pgradients_3/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependency_1Identity;gradients_3/softmax_cross_entropy_with_logits_sg_grad/mul_1G^gradients_3/softmax_cross_entropy_with_logits_sg_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_3/softmax_cross_entropy_with_logits_sg_grad/mul_1

Cgradients_3/softmax_cross_entropy_with_logits_sg/Reshape_grad/ShapeShapeCNN_encoder_cat/catout/Softmax*
T0*
out_type0
ü
Egradients_3/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeReshapeNgradients_3/softmax_cross_entropy_with_logits_sg_grad/tuple/control_dependencyCgradients_3/softmax_cross_entropy_with_logits_sg/Reshape_grad/Shape*
T0*
Tshape0
Ŗ
3gradients_3/CNN_encoder_cat/catout/Softmax_grad/mulMulEgradients_3/softmax_cross_entropy_with_logits_sg/Reshape_grad/ReshapeCNN_encoder_cat/catout/Softmax*
T0
x
Egradients_3/CNN_encoder_cat/catout/Softmax_grad/Sum/reduction_indicesConst*
valueB :
’’’’’’’’’*
dtype0
Ü
3gradients_3/CNN_encoder_cat/catout/Softmax_grad/SumSum3gradients_3/CNN_encoder_cat/catout/Softmax_grad/mulEgradients_3/CNN_encoder_cat/catout/Softmax_grad/Sum/reduction_indices*

Tidx0*
	keep_dims(*
T0
æ
3gradients_3/CNN_encoder_cat/catout/Softmax_grad/subSubEgradients_3/softmax_cross_entropy_with_logits_sg/Reshape_grad/Reshape3gradients_3/CNN_encoder_cat/catout/Softmax_grad/Sum*
T0

5gradients_3/CNN_encoder_cat/catout/Softmax_grad/mul_1Mul3gradients_3/CNN_encoder_cat/catout/Softmax_grad/subCNN_encoder_cat/catout/Softmax*
T0
”
;gradients_3/CNN_encoder_cat/catout/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients_3/CNN_encoder_cat/catout/Softmax_grad/mul_1*
T0*
data_formatNHWC
¾
@gradients_3/CNN_encoder_cat/catout/BiasAdd_grad/tuple/group_depsNoOp<^gradients_3/CNN_encoder_cat/catout/BiasAdd_grad/BiasAddGrad6^gradients_3/CNN_encoder_cat/catout/Softmax_grad/mul_1
”
Hgradients_3/CNN_encoder_cat/catout/BiasAdd_grad/tuple/control_dependencyIdentity5gradients_3/CNN_encoder_cat/catout/Softmax_grad/mul_1A^gradients_3/CNN_encoder_cat/catout/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_3/CNN_encoder_cat/catout/Softmax_grad/mul_1
Æ
Jgradients_3/CNN_encoder_cat/catout/BiasAdd_grad/tuple/control_dependency_1Identity;gradients_3/CNN_encoder_cat/catout/BiasAdd_grad/BiasAddGradA^gradients_3/CNN_encoder_cat/catout/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_3/CNN_encoder_cat/catout/BiasAdd_grad/BiasAddGrad
×
5gradients_3/CNN_encoder_cat/catout/MatMul_grad/MatMulMatMulHgradients_3/CNN_encoder_cat/catout/BiasAdd_grad/tuple/control_dependencyCNN_encoder_cat/catout/W/read*
transpose_b(*
T0*
transpose_a( 
Ś
7gradients_3/CNN_encoder_cat/catout/MatMul_grad/MatMul_1MatMulCNN_encoder_cat/catout/ReshapeHgradients_3/CNN_encoder_cat/catout/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
¹
?gradients_3/CNN_encoder_cat/catout/MatMul_grad/tuple/group_depsNoOp6^gradients_3/CNN_encoder_cat/catout/MatMul_grad/MatMul8^gradients_3/CNN_encoder_cat/catout/MatMul_grad/MatMul_1

Ggradients_3/CNN_encoder_cat/catout/MatMul_grad/tuple/control_dependencyIdentity5gradients_3/CNN_encoder_cat/catout/MatMul_grad/MatMul@^gradients_3/CNN_encoder_cat/catout/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_3/CNN_encoder_cat/catout/MatMul_grad/MatMul
„
Igradients_3/CNN_encoder_cat/catout/MatMul_grad/tuple/control_dependency_1Identity7gradients_3/CNN_encoder_cat/catout/MatMul_grad/MatMul_1@^gradients_3/CNN_encoder_cat/catout/MatMul_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_3/CNN_encoder_cat/catout/MatMul_grad/MatMul_1
|
5gradients_3/CNN_encoder_cat/catout/Reshape_grad/ShapeShape#CNN_encoder_cat/MaxPool2D_2/MaxPool*
T0*
out_type0
Ł
7gradients_3/CNN_encoder_cat/catout/Reshape_grad/ReshapeReshapeGgradients_3/CNN_encoder_cat/catout/MatMul_grad/tuple/control_dependency5gradients_3/CNN_encoder_cat/catout/Reshape_grad/Shape*
T0*
Tshape0
§
@gradients_3/CNN_encoder_cat/MaxPool2D_2/MaxPool_grad/MaxPoolGradMaxPoolGradCNN_encoder_cat/Conv2D_2/Tanh#CNN_encoder_cat/MaxPool2D_2/MaxPool7gradients_3/CNN_encoder_cat/catout/Reshape_grad/Reshape*
ksize
*
paddingSAME*
T0*
data_formatNHWC*
strides

­
7gradients_3/CNN_encoder_cat/Conv2D_2/Tanh_grad/TanhGradTanhGradCNN_encoder_cat/Conv2D_2/Tanh@gradients_3/CNN_encoder_cat/MaxPool2D_2/MaxPool_grad/MaxPoolGrad*
T0
„
=gradients_3/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients_3/CNN_encoder_cat/Conv2D_2/Tanh_grad/TanhGrad*
T0*
data_formatNHWC
Ä
Bgradients_3/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/group_depsNoOp>^gradients_3/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/BiasAddGrad8^gradients_3/CNN_encoder_cat/Conv2D_2/Tanh_grad/TanhGrad
©
Jgradients_3/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/control_dependencyIdentity7gradients_3/CNN_encoder_cat/Conv2D_2/Tanh_grad/TanhGradC^gradients_3/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_3/CNN_encoder_cat/Conv2D_2/Tanh_grad/TanhGrad
·
Lgradients_3/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/control_dependency_1Identity=gradients_3/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/BiasAddGradC^gradients_3/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_3/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/BiasAddGrad
©
7gradients_3/CNN_encoder_cat/Conv2D_2/Conv2D_grad/ShapeNShapeN#CNN_encoder_cat/MaxPool2D_1/MaxPoolCNN_encoder_cat/Conv2D_2/W/read*
T0*
out_type0*
N

Dgradients_3/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput7gradients_3/CNN_encoder_cat/Conv2D_2/Conv2D_grad/ShapeNCNN_encoder_cat/Conv2D_2/W/readJgradients_3/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME

Egradients_3/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter#CNN_encoder_cat/MaxPool2D_1/MaxPool9gradients_3/CNN_encoder_cat/Conv2D_2/Conv2D_grad/ShapeN:1Jgradients_3/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
Ų
Agradients_3/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/group_depsNoOpF^gradients_3/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropFilterE^gradients_3/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropInput
Į
Igradients_3/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/control_dependencyIdentityDgradients_3/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropInputB^gradients_3/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_3/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropInput
Å
Kgradients_3/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/control_dependency_1IdentityEgradients_3/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropFilterB^gradients_3/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_3/CNN_encoder_cat/Conv2D_2/Conv2D_grad/Conv2DBackpropFilter
¹
@gradients_3/CNN_encoder_cat/MaxPool2D_1/MaxPool_grad/MaxPoolGradMaxPoolGradCNN_encoder_cat/Conv2D_1/Tanh#CNN_encoder_cat/MaxPool2D_1/MaxPoolIgradients_3/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*
T0*
data_formatNHWC*
strides

­
7gradients_3/CNN_encoder_cat/Conv2D_1/Tanh_grad/TanhGradTanhGradCNN_encoder_cat/Conv2D_1/Tanh@gradients_3/CNN_encoder_cat/MaxPool2D_1/MaxPool_grad/MaxPoolGrad*
T0
„
=gradients_3/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients_3/CNN_encoder_cat/Conv2D_1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC
Ä
Bgradients_3/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/group_depsNoOp>^gradients_3/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/BiasAddGrad8^gradients_3/CNN_encoder_cat/Conv2D_1/Tanh_grad/TanhGrad
©
Jgradients_3/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/control_dependencyIdentity7gradients_3/CNN_encoder_cat/Conv2D_1/Tanh_grad/TanhGradC^gradients_3/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients_3/CNN_encoder_cat/Conv2D_1/Tanh_grad/TanhGrad
·
Lgradients_3/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/control_dependency_1Identity=gradients_3/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/BiasAddGradC^gradients_3/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_3/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/BiasAddGrad
§
7gradients_3/CNN_encoder_cat/Conv2D_1/Conv2D_grad/ShapeNShapeN!CNN_encoder_cat/MaxPool2D/MaxPoolCNN_encoder_cat/Conv2D_1/W/read*
T0*
out_type0*
N

Dgradients_3/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput7gradients_3/CNN_encoder_cat/Conv2D_1/Conv2D_grad/ShapeNCNN_encoder_cat/Conv2D_1/W/readJgradients_3/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME

Egradients_3/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter!CNN_encoder_cat/MaxPool2D/MaxPool9gradients_3/CNN_encoder_cat/Conv2D_1/Conv2D_grad/ShapeN:1Jgradients_3/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
Ų
Agradients_3/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/group_depsNoOpF^gradients_3/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropFilterE^gradients_3/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropInput
Į
Igradients_3/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/control_dependencyIdentityDgradients_3/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropInputB^gradients_3/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients_3/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropInput
Å
Kgradients_3/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/control_dependency_1IdentityEgradients_3/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropFilterB^gradients_3/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients_3/CNN_encoder_cat/Conv2D_1/Conv2D_grad/Conv2DBackpropFilter
³
>gradients_3/CNN_encoder_cat/MaxPool2D/MaxPool_grad/MaxPoolGradMaxPoolGradCNN_encoder_cat/Conv2D/Tanh!CNN_encoder_cat/MaxPool2D/MaxPoolIgradients_3/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/control_dependency*
ksize
*
paddingSAME*
T0*
data_formatNHWC*
strides

§
5gradients_3/CNN_encoder_cat/Conv2D/Tanh_grad/TanhGradTanhGradCNN_encoder_cat/Conv2D/Tanh>gradients_3/CNN_encoder_cat/MaxPool2D/MaxPool_grad/MaxPoolGrad*
T0
”
;gradients_3/CNN_encoder_cat/Conv2D/BiasAdd_grad/BiasAddGradBiasAddGrad5gradients_3/CNN_encoder_cat/Conv2D/Tanh_grad/TanhGrad*
T0*
data_formatNHWC
¾
@gradients_3/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/group_depsNoOp<^gradients_3/CNN_encoder_cat/Conv2D/BiasAdd_grad/BiasAddGrad6^gradients_3/CNN_encoder_cat/Conv2D/Tanh_grad/TanhGrad
”
Hgradients_3/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/control_dependencyIdentity5gradients_3/CNN_encoder_cat/Conv2D/Tanh_grad/TanhGradA^gradients_3/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/group_deps*
T0*H
_class>
<:loc:@gradients_3/CNN_encoder_cat/Conv2D/Tanh_grad/TanhGrad
Æ
Jgradients_3/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/control_dependency_1Identity;gradients_3/CNN_encoder_cat/Conv2D/BiasAdd_grad/BiasAddGradA^gradients_3/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_3/CNN_encoder_cat/Conv2D/BiasAdd_grad/BiasAddGrad

5gradients_3/CNN_encoder_cat/Conv2D/Conv2D_grad/ShapeNShapeNCNN_encoder_cat/ReshapeCNN_encoder_cat/Conv2D/W/read*
T0*
out_type0*
N

Bgradients_3/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput5gradients_3/CNN_encoder_cat/Conv2D/Conv2D_grad/ShapeNCNN_encoder_cat/Conv2D/W/readHgradients_3/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME

Cgradients_3/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilterCNN_encoder_cat/Reshape7gradients_3/CNN_encoder_cat/Conv2D/Conv2D_grad/ShapeN:1Hgradients_3/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
explicit_paddings
 *
paddingSAME
Ņ
?gradients_3/CNN_encoder_cat/Conv2D/Conv2D_grad/tuple/group_depsNoOpD^gradients_3/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropFilterC^gradients_3/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropInput
¹
Ggradients_3/CNN_encoder_cat/Conv2D/Conv2D_grad/tuple/control_dependencyIdentityBgradients_3/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropInput@^gradients_3/CNN_encoder_cat/Conv2D/Conv2D_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients_3/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropInput
½
Igradients_3/CNN_encoder_cat/Conv2D/Conv2D_grad/tuple/control_dependency_1IdentityCgradients_3/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropFilter@^gradients_3/CNN_encoder_cat/Conv2D/Conv2D_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_3/CNN_encoder_cat/Conv2D/Conv2D_grad/Conv2DBackpropFilter
u
beta1_power_3/initial_valueConst*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
valueB
 *fff?*
dtype0

beta1_power_3
VariableV2*
shape: *
shared_name *+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
dtype0*
	container 
©
beta1_power_3/AssignAssignbeta1_power_3beta1_power_3/initial_value*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(
c
beta1_power_3/readIdentitybeta1_power_3*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W
u
beta2_power_3/initial_valueConst*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
valueB
 *w¾?*
dtype0

beta2_power_3
VariableV2*
shape: *
shared_name *+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
dtype0*
	container 
©
beta2_power_3/AssignAssignbeta2_power_3beta2_power_3/initial_value*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(
c
beta2_power_3/readIdentitybeta2_power_3*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W

1CNN_encoder_cat/Conv2D/W/Adam_6/Initializer/zerosConst*%
valueB *    *+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
dtype0
Ø
CNN_encoder_cat/Conv2D/W/Adam_6
VariableV2*
shape: *
shared_name *+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
dtype0*
	container 
ć
&CNN_encoder_cat/Conv2D/W/Adam_6/AssignAssignCNN_encoder_cat/Conv2D/W/Adam_61CNN_encoder_cat/Conv2D/W/Adam_6/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(

$CNN_encoder_cat/Conv2D/W/Adam_6/readIdentityCNN_encoder_cat/Conv2D/W/Adam_6*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W

1CNN_encoder_cat/Conv2D/W/Adam_7/Initializer/zerosConst*%
valueB *    *+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
dtype0
Ø
CNN_encoder_cat/Conv2D/W/Adam_7
VariableV2*
shape: *
shared_name *+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
dtype0*
	container 
ć
&CNN_encoder_cat/Conv2D/W/Adam_7/AssignAssignCNN_encoder_cat/Conv2D/W/Adam_71CNN_encoder_cat/Conv2D/W/Adam_7/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(

$CNN_encoder_cat/Conv2D/W/Adam_7/readIdentityCNN_encoder_cat/Conv2D/W/Adam_7*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W

1CNN_encoder_cat/Conv2D/b/Adam_6/Initializer/zerosConst*
valueB *    *+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
dtype0

CNN_encoder_cat/Conv2D/b/Adam_6
VariableV2*
shape: *
shared_name *+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
dtype0*
	container 
ć
&CNN_encoder_cat/Conv2D/b/Adam_6/AssignAssignCNN_encoder_cat/Conv2D/b/Adam_61CNN_encoder_cat/Conv2D/b/Adam_6/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
validate_shape(

$CNN_encoder_cat/Conv2D/b/Adam_6/readIdentityCNN_encoder_cat/Conv2D/b/Adam_6*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b

1CNN_encoder_cat/Conv2D/b/Adam_7/Initializer/zerosConst*
valueB *    *+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
dtype0

CNN_encoder_cat/Conv2D/b/Adam_7
VariableV2*
shape: *
shared_name *+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
dtype0*
	container 
ć
&CNN_encoder_cat/Conv2D/b/Adam_7/AssignAssignCNN_encoder_cat/Conv2D/b/Adam_71CNN_encoder_cat/Conv2D/b/Adam_7/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
validate_shape(

$CNN_encoder_cat/Conv2D/b/Adam_7/readIdentityCNN_encoder_cat/Conv2D/b/Adam_7*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b
Æ
CCNN_encoder_cat/Conv2D_1/W/Adam_6/Initializer/zeros/shape_as_tensorConst*%
valueB"              *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0

9CNN_encoder_cat/Conv2D_1/W/Adam_6/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0

3CNN_encoder_cat/Conv2D_1/W/Adam_6/Initializer/zerosFillCCNN_encoder_cat/Conv2D_1/W/Adam_6/Initializer/zeros/shape_as_tensor9CNN_encoder_cat/Conv2D_1/W/Adam_6/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W
¬
!CNN_encoder_cat/Conv2D_1/W/Adam_6
VariableV2*
shape:  *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_1/W/Adam_6/AssignAssign!CNN_encoder_cat/Conv2D_1/W/Adam_63CNN_encoder_cat/Conv2D_1/W/Adam_6/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
validate_shape(

&CNN_encoder_cat/Conv2D_1/W/Adam_6/readIdentity!CNN_encoder_cat/Conv2D_1/W/Adam_6*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W
Æ
CCNN_encoder_cat/Conv2D_1/W/Adam_7/Initializer/zeros/shape_as_tensorConst*%
valueB"              *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0

9CNN_encoder_cat/Conv2D_1/W/Adam_7/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0

3CNN_encoder_cat/Conv2D_1/W/Adam_7/Initializer/zerosFillCCNN_encoder_cat/Conv2D_1/W/Adam_7/Initializer/zeros/shape_as_tensor9CNN_encoder_cat/Conv2D_1/W/Adam_7/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W
¬
!CNN_encoder_cat/Conv2D_1/W/Adam_7
VariableV2*
shape:  *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_1/W/Adam_7/AssignAssign!CNN_encoder_cat/Conv2D_1/W/Adam_73CNN_encoder_cat/Conv2D_1/W/Adam_7/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
validate_shape(

&CNN_encoder_cat/Conv2D_1/W/Adam_7/readIdentity!CNN_encoder_cat/Conv2D_1/W/Adam_7*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W

3CNN_encoder_cat/Conv2D_1/b/Adam_6/Initializer/zerosConst*
valueB *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
dtype0
 
!CNN_encoder_cat/Conv2D_1/b/Adam_6
VariableV2*
shape: *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_1/b/Adam_6/AssignAssign!CNN_encoder_cat/Conv2D_1/b/Adam_63CNN_encoder_cat/Conv2D_1/b/Adam_6/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
validate_shape(

&CNN_encoder_cat/Conv2D_1/b/Adam_6/readIdentity!CNN_encoder_cat/Conv2D_1/b/Adam_6*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b

3CNN_encoder_cat/Conv2D_1/b/Adam_7/Initializer/zerosConst*
valueB *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
dtype0
 
!CNN_encoder_cat/Conv2D_1/b/Adam_7
VariableV2*
shape: *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_1/b/Adam_7/AssignAssign!CNN_encoder_cat/Conv2D_1/b/Adam_73CNN_encoder_cat/Conv2D_1/b/Adam_7/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
validate_shape(

&CNN_encoder_cat/Conv2D_1/b/Adam_7/readIdentity!CNN_encoder_cat/Conv2D_1/b/Adam_7*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b
Æ
CCNN_encoder_cat/Conv2D_2/W/Adam_6/Initializer/zeros/shape_as_tensorConst*%
valueB"              *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0

9CNN_encoder_cat/Conv2D_2/W/Adam_6/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0

3CNN_encoder_cat/Conv2D_2/W/Adam_6/Initializer/zerosFillCCNN_encoder_cat/Conv2D_2/W/Adam_6/Initializer/zeros/shape_as_tensor9CNN_encoder_cat/Conv2D_2/W/Adam_6/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W
¬
!CNN_encoder_cat/Conv2D_2/W/Adam_6
VariableV2*
shape:  *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_2/W/Adam_6/AssignAssign!CNN_encoder_cat/Conv2D_2/W/Adam_63CNN_encoder_cat/Conv2D_2/W/Adam_6/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
validate_shape(

&CNN_encoder_cat/Conv2D_2/W/Adam_6/readIdentity!CNN_encoder_cat/Conv2D_2/W/Adam_6*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W
Æ
CCNN_encoder_cat/Conv2D_2/W/Adam_7/Initializer/zeros/shape_as_tensorConst*%
valueB"              *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0

9CNN_encoder_cat/Conv2D_2/W/Adam_7/Initializer/zeros/ConstConst*
valueB
 *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0

3CNN_encoder_cat/Conv2D_2/W/Adam_7/Initializer/zerosFillCCNN_encoder_cat/Conv2D_2/W/Adam_7/Initializer/zeros/shape_as_tensor9CNN_encoder_cat/Conv2D_2/W/Adam_7/Initializer/zeros/Const*
T0*

index_type0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W
¬
!CNN_encoder_cat/Conv2D_2/W/Adam_7
VariableV2*
shape:  *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_2/W/Adam_7/AssignAssign!CNN_encoder_cat/Conv2D_2/W/Adam_73CNN_encoder_cat/Conv2D_2/W/Adam_7/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
validate_shape(

&CNN_encoder_cat/Conv2D_2/W/Adam_7/readIdentity!CNN_encoder_cat/Conv2D_2/W/Adam_7*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W

3CNN_encoder_cat/Conv2D_2/b/Adam_6/Initializer/zerosConst*
valueB *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
dtype0
 
!CNN_encoder_cat/Conv2D_2/b/Adam_6
VariableV2*
shape: *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_2/b/Adam_6/AssignAssign!CNN_encoder_cat/Conv2D_2/b/Adam_63CNN_encoder_cat/Conv2D_2/b/Adam_6/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
validate_shape(

&CNN_encoder_cat/Conv2D_2/b/Adam_6/readIdentity!CNN_encoder_cat/Conv2D_2/b/Adam_6*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b

3CNN_encoder_cat/Conv2D_2/b/Adam_7/Initializer/zerosConst*
valueB *    *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
dtype0
 
!CNN_encoder_cat/Conv2D_2/b/Adam_7
VariableV2*
shape: *
shared_name *-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
dtype0*
	container 
ė
(CNN_encoder_cat/Conv2D_2/b/Adam_7/AssignAssign!CNN_encoder_cat/Conv2D_2/b/Adam_73CNN_encoder_cat/Conv2D_2/b/Adam_7/Initializer/zeros*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
validate_shape(

&CNN_encoder_cat/Conv2D_2/b/Adam_7/readIdentity!CNN_encoder_cat/Conv2D_2/b/Adam_7*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b
£
ACNN_encoder_cat/catout/W/Adam_6/Initializer/zeros/shape_as_tensorConst*
valueB"     *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0

7CNN_encoder_cat/catout/W/Adam_6/Initializer/zeros/ConstConst*
valueB
 *    *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0
ż
1CNN_encoder_cat/catout/W/Adam_6/Initializer/zerosFillACNN_encoder_cat/catout/W/Adam_6/Initializer/zeros/shape_as_tensor7CNN_encoder_cat/catout/W/Adam_6/Initializer/zeros/Const*
T0*

index_type0*+
_class!
loc:@CNN_encoder_cat/catout/W
”
CNN_encoder_cat/catout/W/Adam_6
VariableV2*
shape:	*
shared_name *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0*
	container 
ć
&CNN_encoder_cat/catout/W/Adam_6/AssignAssignCNN_encoder_cat/catout/W/Adam_61CNN_encoder_cat/catout/W/Adam_6/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W*
validate_shape(

$CNN_encoder_cat/catout/W/Adam_6/readIdentityCNN_encoder_cat/catout/W/Adam_6*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W
£
ACNN_encoder_cat/catout/W/Adam_7/Initializer/zeros/shape_as_tensorConst*
valueB"     *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0

7CNN_encoder_cat/catout/W/Adam_7/Initializer/zeros/ConstConst*
valueB
 *    *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0
ż
1CNN_encoder_cat/catout/W/Adam_7/Initializer/zerosFillACNN_encoder_cat/catout/W/Adam_7/Initializer/zeros/shape_as_tensor7CNN_encoder_cat/catout/W/Adam_7/Initializer/zeros/Const*
T0*

index_type0*+
_class!
loc:@CNN_encoder_cat/catout/W
”
CNN_encoder_cat/catout/W/Adam_7
VariableV2*
shape:	*
shared_name *+
_class!
loc:@CNN_encoder_cat/catout/W*
dtype0*
	container 
ć
&CNN_encoder_cat/catout/W/Adam_7/AssignAssignCNN_encoder_cat/catout/W/Adam_71CNN_encoder_cat/catout/W/Adam_7/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W*
validate_shape(

$CNN_encoder_cat/catout/W/Adam_7/readIdentityCNN_encoder_cat/catout/W/Adam_7*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W

1CNN_encoder_cat/catout/b/Adam_6/Initializer/zerosConst*
valueB*    *+
_class!
loc:@CNN_encoder_cat/catout/b*
dtype0

CNN_encoder_cat/catout/b/Adam_6
VariableV2*
shape:*
shared_name *+
_class!
loc:@CNN_encoder_cat/catout/b*
dtype0*
	container 
ć
&CNN_encoder_cat/catout/b/Adam_6/AssignAssignCNN_encoder_cat/catout/b/Adam_61CNN_encoder_cat/catout/b/Adam_6/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b*
validate_shape(

$CNN_encoder_cat/catout/b/Adam_6/readIdentityCNN_encoder_cat/catout/b/Adam_6*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b

1CNN_encoder_cat/catout/b/Adam_7/Initializer/zerosConst*
valueB*    *+
_class!
loc:@CNN_encoder_cat/catout/b*
dtype0

CNN_encoder_cat/catout/b/Adam_7
VariableV2*
shape:*
shared_name *+
_class!
loc:@CNN_encoder_cat/catout/b*
dtype0*
	container 
ć
&CNN_encoder_cat/catout/b/Adam_7/AssignAssignCNN_encoder_cat/catout/b/Adam_71CNN_encoder_cat/catout/b/Adam_7/Initializer/zeros*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b*
validate_shape(

$CNN_encoder_cat/catout/b/Adam_7/readIdentityCNN_encoder_cat/catout/b/Adam_7*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b
A
Adam_3/learning_rateConst*
valueB
 *·Q8*
dtype0
9
Adam_3/beta1Const*
valueB
 *fff?*
dtype0
9
Adam_3/beta2Const*
valueB
 *w¾?*
dtype0
;
Adam_3/epsilonConst*
valueB
 *wĢ+2*
dtype0
«
0Adam_3/update_CNN_encoder_cat/Conv2D/W/ApplyAdam	ApplyAdamCNN_encoder_cat/Conv2D/WCNN_encoder_cat/Conv2D/W/Adam_6CNN_encoder_cat/Conv2D/W/Adam_7beta1_power_3/readbeta2_power_3/readAdam_3/learning_rateAdam_3/beta1Adam_3/beta2Adam_3/epsilonIgradients_3/CNN_encoder_cat/Conv2D/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
use_nesterov( 
¬
0Adam_3/update_CNN_encoder_cat/Conv2D/b/ApplyAdam	ApplyAdamCNN_encoder_cat/Conv2D/bCNN_encoder_cat/Conv2D/b/Adam_6CNN_encoder_cat/Conv2D/b/Adam_7beta1_power_3/readbeta2_power_3/readAdam_3/learning_rateAdam_3/beta1Adam_3/beta2Adam_3/epsilonJgradients_3/CNN_encoder_cat/Conv2D/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
use_nesterov( 
·
2Adam_3/update_CNN_encoder_cat/Conv2D_1/W/ApplyAdam	ApplyAdamCNN_encoder_cat/Conv2D_1/W!CNN_encoder_cat/Conv2D_1/W/Adam_6!CNN_encoder_cat/Conv2D_1/W/Adam_7beta1_power_3/readbeta2_power_3/readAdam_3/learning_rateAdam_3/beta1Adam_3/beta2Adam_3/epsilonKgradients_3/CNN_encoder_cat/Conv2D_1/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
use_nesterov( 
ø
2Adam_3/update_CNN_encoder_cat/Conv2D_1/b/ApplyAdam	ApplyAdamCNN_encoder_cat/Conv2D_1/b!CNN_encoder_cat/Conv2D_1/b/Adam_6!CNN_encoder_cat/Conv2D_1/b/Adam_7beta1_power_3/readbeta2_power_3/readAdam_3/learning_rateAdam_3/beta1Adam_3/beta2Adam_3/epsilonLgradients_3/CNN_encoder_cat/Conv2D_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
use_nesterov( 
·
2Adam_3/update_CNN_encoder_cat/Conv2D_2/W/ApplyAdam	ApplyAdamCNN_encoder_cat/Conv2D_2/W!CNN_encoder_cat/Conv2D_2/W/Adam_6!CNN_encoder_cat/Conv2D_2/W/Adam_7beta1_power_3/readbeta2_power_3/readAdam_3/learning_rateAdam_3/beta1Adam_3/beta2Adam_3/epsilonKgradients_3/CNN_encoder_cat/Conv2D_2/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
use_nesterov( 
ø
2Adam_3/update_CNN_encoder_cat/Conv2D_2/b/ApplyAdam	ApplyAdamCNN_encoder_cat/Conv2D_2/b!CNN_encoder_cat/Conv2D_2/b/Adam_6!CNN_encoder_cat/Conv2D_2/b/Adam_7beta1_power_3/readbeta2_power_3/readAdam_3/learning_rateAdam_3/beta1Adam_3/beta2Adam_3/epsilonLgradients_3/CNN_encoder_cat/Conv2D_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
use_nesterov( 
«
0Adam_3/update_CNN_encoder_cat/catout/W/ApplyAdam	ApplyAdamCNN_encoder_cat/catout/WCNN_encoder_cat/catout/W/Adam_6CNN_encoder_cat/catout/W/Adam_7beta1_power_3/readbeta2_power_3/readAdam_3/learning_rateAdam_3/beta1Adam_3/beta2Adam_3/epsilonIgradients_3/CNN_encoder_cat/catout/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*+
_class!
loc:@CNN_encoder_cat/catout/W*
use_nesterov( 
¬
0Adam_3/update_CNN_encoder_cat/catout/b/ApplyAdam	ApplyAdamCNN_encoder_cat/catout/bCNN_encoder_cat/catout/b/Adam_6CNN_encoder_cat/catout/b/Adam_7beta1_power_3/readbeta2_power_3/readAdam_3/learning_rateAdam_3/beta1Adam_3/beta2Adam_3/epsilonJgradients_3/CNN_encoder_cat/catout/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*+
_class!
loc:@CNN_encoder_cat/catout/b*
use_nesterov( 


Adam_3/mulMulbeta1_power_3/readAdam_3/beta11^Adam_3/update_CNN_encoder_cat/Conv2D/W/ApplyAdam1^Adam_3/update_CNN_encoder_cat/Conv2D/b/ApplyAdam3^Adam_3/update_CNN_encoder_cat/Conv2D_1/W/ApplyAdam3^Adam_3/update_CNN_encoder_cat/Conv2D_1/b/ApplyAdam3^Adam_3/update_CNN_encoder_cat/Conv2D_2/W/ApplyAdam3^Adam_3/update_CNN_encoder_cat/Conv2D_2/b/ApplyAdam1^Adam_3/update_CNN_encoder_cat/catout/W/ApplyAdam1^Adam_3/update_CNN_encoder_cat/catout/b/ApplyAdam*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W

Adam_3/AssignAssignbeta1_power_3
Adam_3/mul*
use_locking( *
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(

Adam_3/mul_1Mulbeta2_power_3/readAdam_3/beta21^Adam_3/update_CNN_encoder_cat/Conv2D/W/ApplyAdam1^Adam_3/update_CNN_encoder_cat/Conv2D/b/ApplyAdam3^Adam_3/update_CNN_encoder_cat/Conv2D_1/W/ApplyAdam3^Adam_3/update_CNN_encoder_cat/Conv2D_1/b/ApplyAdam3^Adam_3/update_CNN_encoder_cat/Conv2D_2/W/ApplyAdam3^Adam_3/update_CNN_encoder_cat/Conv2D_2/b/ApplyAdam1^Adam_3/update_CNN_encoder_cat/catout/W/ApplyAdam1^Adam_3/update_CNN_encoder_cat/catout/b/ApplyAdam*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W

Adam_3/Assign_1Assignbeta2_power_3Adam_3/mul_1*
use_locking( *
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(
Š
Adam_3NoOp^Adam_3/Assign^Adam_3/Assign_11^Adam_3/update_CNN_encoder_cat/Conv2D/W/ApplyAdam1^Adam_3/update_CNN_encoder_cat/Conv2D/b/ApplyAdam3^Adam_3/update_CNN_encoder_cat/Conv2D_1/W/ApplyAdam3^Adam_3/update_CNN_encoder_cat/Conv2D_1/b/ApplyAdam3^Adam_3/update_CNN_encoder_cat/Conv2D_2/W/ApplyAdam3^Adam_3/update_CNN_encoder_cat/Conv2D_2/b/ApplyAdam1^Adam_3/update_CNN_encoder_cat/catout/W/ApplyAdam1^Adam_3/update_CNN_encoder_cat/catout/b/ApplyAdam
K
latent_variable_1Placeholder*
shape:’’’’’’’’’2*
dtype0
A
save/filename/inputConst*
valueB Bmodel*
dtype0
V
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0
M

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0
&
save/SaveV2/tensor_namesConst*Ū%
valueŃ%BĪ%§BCNN_decoder/Conv2D/WBCNN_decoder/Conv2D/W/AdamBCNN_decoder/Conv2D/W/Adam_1BCNN_decoder/Conv2D/bBCNN_decoder/Conv2D/b/AdamBCNN_decoder/Conv2D/b/Adam_1BCNN_decoder/Conv2D_1/WBCNN_decoder/Conv2D_1/W/AdamBCNN_decoder/Conv2D_1/W/Adam_1BCNN_decoder/Conv2D_1/bBCNN_decoder/Conv2D_1/b/AdamBCNN_decoder/Conv2D_1/b/Adam_1BCNN_decoder/Conv2D_2/WBCNN_decoder/Conv2D_2/W/AdamBCNN_decoder/Conv2D_2/W/Adam_1BCNN_decoder/Conv2D_2/bBCNN_decoder/Conv2D_2/b/AdamBCNN_decoder/Conv2D_2/b/Adam_1BCNN_decoder/FullyConnected/WB!CNN_decoder/FullyConnected/W/AdamB#CNN_decoder/FullyConnected/W/Adam_1BCNN_decoder/FullyConnected/bB!CNN_decoder/FullyConnected/b/AdamB#CNN_decoder/FullyConnected/b/Adam_1BCNN_decoder/sigout/WBCNN_decoder/sigout/W/AdamBCNN_decoder/sigout/W/Adam_1BCNN_decoder/sigout/bBCNN_decoder/sigout/b/AdamBCNN_decoder/sigout/b/Adam_1BCNN_decoder/sigout_1/WBCNN_decoder/sigout_1/W/AdamBCNN_decoder/sigout_1/W/Adam_1BCNN_decoder/sigout_1/bBCNN_decoder/sigout_1/b/AdamBCNN_decoder/sigout_1/b/Adam_1BCNN_encoder_cat/Conv2D/WBCNN_encoder_cat/Conv2D/W/AdamBCNN_encoder_cat/Conv2D/W/Adam_1BCNN_encoder_cat/Conv2D/W/Adam_2BCNN_encoder_cat/Conv2D/W/Adam_3BCNN_encoder_cat/Conv2D/W/Adam_4BCNN_encoder_cat/Conv2D/W/Adam_5BCNN_encoder_cat/Conv2D/W/Adam_6BCNN_encoder_cat/Conv2D/W/Adam_7BCNN_encoder_cat/Conv2D/bBCNN_encoder_cat/Conv2D/b/AdamBCNN_encoder_cat/Conv2D/b/Adam_1BCNN_encoder_cat/Conv2D/b/Adam_2BCNN_encoder_cat/Conv2D/b/Adam_3BCNN_encoder_cat/Conv2D/b/Adam_4BCNN_encoder_cat/Conv2D/b/Adam_5BCNN_encoder_cat/Conv2D/b/Adam_6BCNN_encoder_cat/Conv2D/b/Adam_7BCNN_encoder_cat/Conv2D_1/WBCNN_encoder_cat/Conv2D_1/W/AdamB!CNN_encoder_cat/Conv2D_1/W/Adam_1B!CNN_encoder_cat/Conv2D_1/W/Adam_2B!CNN_encoder_cat/Conv2D_1/W/Adam_3B!CNN_encoder_cat/Conv2D_1/W/Adam_4B!CNN_encoder_cat/Conv2D_1/W/Adam_5B!CNN_encoder_cat/Conv2D_1/W/Adam_6B!CNN_encoder_cat/Conv2D_1/W/Adam_7BCNN_encoder_cat/Conv2D_1/bBCNN_encoder_cat/Conv2D_1/b/AdamB!CNN_encoder_cat/Conv2D_1/b/Adam_1B!CNN_encoder_cat/Conv2D_1/b/Adam_2B!CNN_encoder_cat/Conv2D_1/b/Adam_3B!CNN_encoder_cat/Conv2D_1/b/Adam_4B!CNN_encoder_cat/Conv2D_1/b/Adam_5B!CNN_encoder_cat/Conv2D_1/b/Adam_6B!CNN_encoder_cat/Conv2D_1/b/Adam_7BCNN_encoder_cat/Conv2D_2/WBCNN_encoder_cat/Conv2D_2/W/AdamB!CNN_encoder_cat/Conv2D_2/W/Adam_1B!CNN_encoder_cat/Conv2D_2/W/Adam_2B!CNN_encoder_cat/Conv2D_2/W/Adam_3B!CNN_encoder_cat/Conv2D_2/W/Adam_4B!CNN_encoder_cat/Conv2D_2/W/Adam_5B!CNN_encoder_cat/Conv2D_2/W/Adam_6B!CNN_encoder_cat/Conv2D_2/W/Adam_7BCNN_encoder_cat/Conv2D_2/bBCNN_encoder_cat/Conv2D_2/b/AdamB!CNN_encoder_cat/Conv2D_2/b/Adam_1B!CNN_encoder_cat/Conv2D_2/b/Adam_2B!CNN_encoder_cat/Conv2D_2/b/Adam_3B!CNN_encoder_cat/Conv2D_2/b/Adam_4B!CNN_encoder_cat/Conv2D_2/b/Adam_5B!CNN_encoder_cat/Conv2D_2/b/Adam_6B!CNN_encoder_cat/Conv2D_2/b/Adam_7BCNN_encoder_cat/catout/WBCNN_encoder_cat/catout/W/AdamBCNN_encoder_cat/catout/W/Adam_1BCNN_encoder_cat/catout/W/Adam_2BCNN_encoder_cat/catout/W/Adam_3BCNN_encoder_cat/catout/W/Adam_4BCNN_encoder_cat/catout/W/Adam_5BCNN_encoder_cat/catout/W/Adam_6BCNN_encoder_cat/catout/W/Adam_7BCNN_encoder_cat/catout/bBCNN_encoder_cat/catout/b/AdamBCNN_encoder_cat/catout/b/Adam_1BCNN_encoder_cat/catout/b/Adam_2BCNN_encoder_cat/catout/b/Adam_3BCNN_encoder_cat/catout/b/Adam_4BCNN_encoder_cat/catout/b/Adam_5BCNN_encoder_cat/catout/b/Adam_6BCNN_encoder_cat/catout/b/Adam_7BCNN_encoder_cat/zout/WBCNN_encoder_cat/zout/W/AdamBCNN_encoder_cat/zout/W/Adam_1BCNN_encoder_cat/zout/W/Adam_2BCNN_encoder_cat/zout/W/Adam_3BCNN_encoder_cat/zout/W/Adam_4BCNN_encoder_cat/zout/W/Adam_5BCNN_encoder_cat/zout/bBCNN_encoder_cat/zout/b/AdamBCNN_encoder_cat/zout/b/Adam_1BCNN_encoder_cat/zout/b/Adam_2BCNN_encoder_cat/zout/b/Adam_3BCNN_encoder_cat/zout/b/Adam_4BCNN_encoder_cat/zout/b/Adam_5Bbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta1_power_3Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bbeta2_power_3Bdiscriminator/b0Bdiscriminator/b0/AdamBdiscriminator/b0/Adam_1Bdiscriminator/b1Bdiscriminator/b1/AdamBdiscriminator/b1/Adam_1Bdiscriminator/boBdiscriminator/bo/AdamBdiscriminator/bo/Adam_1Bdiscriminator/w0Bdiscriminator/w0/AdamBdiscriminator/w0/Adam_1Bdiscriminator/w1Bdiscriminator/w1/AdamBdiscriminator/w1/Adam_1Bdiscriminator/woBdiscriminator/wo/AdamBdiscriminator/wo/Adam_1Bdiscriminator_cat/b0Bdiscriminator_cat/b0/AdamBdiscriminator_cat/b0/Adam_1Bdiscriminator_cat/b1Bdiscriminator_cat/b1/AdamBdiscriminator_cat/b1/Adam_1Bdiscriminator_cat/boBdiscriminator_cat/bo/AdamBdiscriminator_cat/bo/Adam_1Bdiscriminator_cat/w0Bdiscriminator_cat/w0/AdamBdiscriminator_cat/w0/Adam_1Bdiscriminator_cat/w1Bdiscriminator_cat/w1/AdamBdiscriminator_cat/w1/Adam_1Bdiscriminator_cat/woBdiscriminator_cat/wo/AdamBdiscriminator_cat/wo/Adam_1Bis_training*
dtype0

save/SaveV2/shape_and_slicesConst*ä
valueŚB×§B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
Ł'
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesCNN_decoder/Conv2D/WCNN_decoder/Conv2D/W/AdamCNN_decoder/Conv2D/W/Adam_1CNN_decoder/Conv2D/bCNN_decoder/Conv2D/b/AdamCNN_decoder/Conv2D/b/Adam_1CNN_decoder/Conv2D_1/WCNN_decoder/Conv2D_1/W/AdamCNN_decoder/Conv2D_1/W/Adam_1CNN_decoder/Conv2D_1/bCNN_decoder/Conv2D_1/b/AdamCNN_decoder/Conv2D_1/b/Adam_1CNN_decoder/Conv2D_2/WCNN_decoder/Conv2D_2/W/AdamCNN_decoder/Conv2D_2/W/Adam_1CNN_decoder/Conv2D_2/bCNN_decoder/Conv2D_2/b/AdamCNN_decoder/Conv2D_2/b/Adam_1CNN_decoder/FullyConnected/W!CNN_decoder/FullyConnected/W/Adam#CNN_decoder/FullyConnected/W/Adam_1CNN_decoder/FullyConnected/b!CNN_decoder/FullyConnected/b/Adam#CNN_decoder/FullyConnected/b/Adam_1CNN_decoder/sigout/WCNN_decoder/sigout/W/AdamCNN_decoder/sigout/W/Adam_1CNN_decoder/sigout/bCNN_decoder/sigout/b/AdamCNN_decoder/sigout/b/Adam_1CNN_decoder/sigout_1/WCNN_decoder/sigout_1/W/AdamCNN_decoder/sigout_1/W/Adam_1CNN_decoder/sigout_1/bCNN_decoder/sigout_1/b/AdamCNN_decoder/sigout_1/b/Adam_1CNN_encoder_cat/Conv2D/WCNN_encoder_cat/Conv2D/W/AdamCNN_encoder_cat/Conv2D/W/Adam_1CNN_encoder_cat/Conv2D/W/Adam_2CNN_encoder_cat/Conv2D/W/Adam_3CNN_encoder_cat/Conv2D/W/Adam_4CNN_encoder_cat/Conv2D/W/Adam_5CNN_encoder_cat/Conv2D/W/Adam_6CNN_encoder_cat/Conv2D/W/Adam_7CNN_encoder_cat/Conv2D/bCNN_encoder_cat/Conv2D/b/AdamCNN_encoder_cat/Conv2D/b/Adam_1CNN_encoder_cat/Conv2D/b/Adam_2CNN_encoder_cat/Conv2D/b/Adam_3CNN_encoder_cat/Conv2D/b/Adam_4CNN_encoder_cat/Conv2D/b/Adam_5CNN_encoder_cat/Conv2D/b/Adam_6CNN_encoder_cat/Conv2D/b/Adam_7CNN_encoder_cat/Conv2D_1/WCNN_encoder_cat/Conv2D_1/W/Adam!CNN_encoder_cat/Conv2D_1/W/Adam_1!CNN_encoder_cat/Conv2D_1/W/Adam_2!CNN_encoder_cat/Conv2D_1/W/Adam_3!CNN_encoder_cat/Conv2D_1/W/Adam_4!CNN_encoder_cat/Conv2D_1/W/Adam_5!CNN_encoder_cat/Conv2D_1/W/Adam_6!CNN_encoder_cat/Conv2D_1/W/Adam_7CNN_encoder_cat/Conv2D_1/bCNN_encoder_cat/Conv2D_1/b/Adam!CNN_encoder_cat/Conv2D_1/b/Adam_1!CNN_encoder_cat/Conv2D_1/b/Adam_2!CNN_encoder_cat/Conv2D_1/b/Adam_3!CNN_encoder_cat/Conv2D_1/b/Adam_4!CNN_encoder_cat/Conv2D_1/b/Adam_5!CNN_encoder_cat/Conv2D_1/b/Adam_6!CNN_encoder_cat/Conv2D_1/b/Adam_7CNN_encoder_cat/Conv2D_2/WCNN_encoder_cat/Conv2D_2/W/Adam!CNN_encoder_cat/Conv2D_2/W/Adam_1!CNN_encoder_cat/Conv2D_2/W/Adam_2!CNN_encoder_cat/Conv2D_2/W/Adam_3!CNN_encoder_cat/Conv2D_2/W/Adam_4!CNN_encoder_cat/Conv2D_2/W/Adam_5!CNN_encoder_cat/Conv2D_2/W/Adam_6!CNN_encoder_cat/Conv2D_2/W/Adam_7CNN_encoder_cat/Conv2D_2/bCNN_encoder_cat/Conv2D_2/b/Adam!CNN_encoder_cat/Conv2D_2/b/Adam_1!CNN_encoder_cat/Conv2D_2/b/Adam_2!CNN_encoder_cat/Conv2D_2/b/Adam_3!CNN_encoder_cat/Conv2D_2/b/Adam_4!CNN_encoder_cat/Conv2D_2/b/Adam_5!CNN_encoder_cat/Conv2D_2/b/Adam_6!CNN_encoder_cat/Conv2D_2/b/Adam_7CNN_encoder_cat/catout/WCNN_encoder_cat/catout/W/AdamCNN_encoder_cat/catout/W/Adam_1CNN_encoder_cat/catout/W/Adam_2CNN_encoder_cat/catout/W/Adam_3CNN_encoder_cat/catout/W/Adam_4CNN_encoder_cat/catout/W/Adam_5CNN_encoder_cat/catout/W/Adam_6CNN_encoder_cat/catout/W/Adam_7CNN_encoder_cat/catout/bCNN_encoder_cat/catout/b/AdamCNN_encoder_cat/catout/b/Adam_1CNN_encoder_cat/catout/b/Adam_2CNN_encoder_cat/catout/b/Adam_3CNN_encoder_cat/catout/b/Adam_4CNN_encoder_cat/catout/b/Adam_5CNN_encoder_cat/catout/b/Adam_6CNN_encoder_cat/catout/b/Adam_7CNN_encoder_cat/zout/WCNN_encoder_cat/zout/W/AdamCNN_encoder_cat/zout/W/Adam_1CNN_encoder_cat/zout/W/Adam_2CNN_encoder_cat/zout/W/Adam_3CNN_encoder_cat/zout/W/Adam_4CNN_encoder_cat/zout/W/Adam_5CNN_encoder_cat/zout/bCNN_encoder_cat/zout/b/AdamCNN_encoder_cat/zout/b/Adam_1CNN_encoder_cat/zout/b/Adam_2CNN_encoder_cat/zout/b/Adam_3CNN_encoder_cat/zout/b/Adam_4CNN_encoder_cat/zout/b/Adam_5beta1_powerbeta1_power_1beta1_power_2beta1_power_3beta2_powerbeta2_power_1beta2_power_2beta2_power_3discriminator/b0discriminator/b0/Adamdiscriminator/b0/Adam_1discriminator/b1discriminator/b1/Adamdiscriminator/b1/Adam_1discriminator/bodiscriminator/bo/Adamdiscriminator/bo/Adam_1discriminator/w0discriminator/w0/Adamdiscriminator/w0/Adam_1discriminator/w1discriminator/w1/Adamdiscriminator/w1/Adam_1discriminator/wodiscriminator/wo/Adamdiscriminator/wo/Adam_1discriminator_cat/b0discriminator_cat/b0/Adamdiscriminator_cat/b0/Adam_1discriminator_cat/b1discriminator_cat/b1/Adamdiscriminator_cat/b1/Adam_1discriminator_cat/bodiscriminator_cat/bo/Adamdiscriminator_cat/bo/Adam_1discriminator_cat/w0discriminator_cat/w0/Adamdiscriminator_cat/w0/Adam_1discriminator_cat/w1discriminator_cat/w1/Adamdiscriminator_cat/w1/Adam_1discriminator_cat/wodiscriminator_cat/wo/Adamdiscriminator_cat/wo/Adam_1is_training*ø
dtypes­
Ŗ2§

e
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const
&
save/RestoreV2/tensor_namesConst"/device:CPU:0*Ū%
valueŃ%BĪ%§BCNN_decoder/Conv2D/WBCNN_decoder/Conv2D/W/AdamBCNN_decoder/Conv2D/W/Adam_1BCNN_decoder/Conv2D/bBCNN_decoder/Conv2D/b/AdamBCNN_decoder/Conv2D/b/Adam_1BCNN_decoder/Conv2D_1/WBCNN_decoder/Conv2D_1/W/AdamBCNN_decoder/Conv2D_1/W/Adam_1BCNN_decoder/Conv2D_1/bBCNN_decoder/Conv2D_1/b/AdamBCNN_decoder/Conv2D_1/b/Adam_1BCNN_decoder/Conv2D_2/WBCNN_decoder/Conv2D_2/W/AdamBCNN_decoder/Conv2D_2/W/Adam_1BCNN_decoder/Conv2D_2/bBCNN_decoder/Conv2D_2/b/AdamBCNN_decoder/Conv2D_2/b/Adam_1BCNN_decoder/FullyConnected/WB!CNN_decoder/FullyConnected/W/AdamB#CNN_decoder/FullyConnected/W/Adam_1BCNN_decoder/FullyConnected/bB!CNN_decoder/FullyConnected/b/AdamB#CNN_decoder/FullyConnected/b/Adam_1BCNN_decoder/sigout/WBCNN_decoder/sigout/W/AdamBCNN_decoder/sigout/W/Adam_1BCNN_decoder/sigout/bBCNN_decoder/sigout/b/AdamBCNN_decoder/sigout/b/Adam_1BCNN_decoder/sigout_1/WBCNN_decoder/sigout_1/W/AdamBCNN_decoder/sigout_1/W/Adam_1BCNN_decoder/sigout_1/bBCNN_decoder/sigout_1/b/AdamBCNN_decoder/sigout_1/b/Adam_1BCNN_encoder_cat/Conv2D/WBCNN_encoder_cat/Conv2D/W/AdamBCNN_encoder_cat/Conv2D/W/Adam_1BCNN_encoder_cat/Conv2D/W/Adam_2BCNN_encoder_cat/Conv2D/W/Adam_3BCNN_encoder_cat/Conv2D/W/Adam_4BCNN_encoder_cat/Conv2D/W/Adam_5BCNN_encoder_cat/Conv2D/W/Adam_6BCNN_encoder_cat/Conv2D/W/Adam_7BCNN_encoder_cat/Conv2D/bBCNN_encoder_cat/Conv2D/b/AdamBCNN_encoder_cat/Conv2D/b/Adam_1BCNN_encoder_cat/Conv2D/b/Adam_2BCNN_encoder_cat/Conv2D/b/Adam_3BCNN_encoder_cat/Conv2D/b/Adam_4BCNN_encoder_cat/Conv2D/b/Adam_5BCNN_encoder_cat/Conv2D/b/Adam_6BCNN_encoder_cat/Conv2D/b/Adam_7BCNN_encoder_cat/Conv2D_1/WBCNN_encoder_cat/Conv2D_1/W/AdamB!CNN_encoder_cat/Conv2D_1/W/Adam_1B!CNN_encoder_cat/Conv2D_1/W/Adam_2B!CNN_encoder_cat/Conv2D_1/W/Adam_3B!CNN_encoder_cat/Conv2D_1/W/Adam_4B!CNN_encoder_cat/Conv2D_1/W/Adam_5B!CNN_encoder_cat/Conv2D_1/W/Adam_6B!CNN_encoder_cat/Conv2D_1/W/Adam_7BCNN_encoder_cat/Conv2D_1/bBCNN_encoder_cat/Conv2D_1/b/AdamB!CNN_encoder_cat/Conv2D_1/b/Adam_1B!CNN_encoder_cat/Conv2D_1/b/Adam_2B!CNN_encoder_cat/Conv2D_1/b/Adam_3B!CNN_encoder_cat/Conv2D_1/b/Adam_4B!CNN_encoder_cat/Conv2D_1/b/Adam_5B!CNN_encoder_cat/Conv2D_1/b/Adam_6B!CNN_encoder_cat/Conv2D_1/b/Adam_7BCNN_encoder_cat/Conv2D_2/WBCNN_encoder_cat/Conv2D_2/W/AdamB!CNN_encoder_cat/Conv2D_2/W/Adam_1B!CNN_encoder_cat/Conv2D_2/W/Adam_2B!CNN_encoder_cat/Conv2D_2/W/Adam_3B!CNN_encoder_cat/Conv2D_2/W/Adam_4B!CNN_encoder_cat/Conv2D_2/W/Adam_5B!CNN_encoder_cat/Conv2D_2/W/Adam_6B!CNN_encoder_cat/Conv2D_2/W/Adam_7BCNN_encoder_cat/Conv2D_2/bBCNN_encoder_cat/Conv2D_2/b/AdamB!CNN_encoder_cat/Conv2D_2/b/Adam_1B!CNN_encoder_cat/Conv2D_2/b/Adam_2B!CNN_encoder_cat/Conv2D_2/b/Adam_3B!CNN_encoder_cat/Conv2D_2/b/Adam_4B!CNN_encoder_cat/Conv2D_2/b/Adam_5B!CNN_encoder_cat/Conv2D_2/b/Adam_6B!CNN_encoder_cat/Conv2D_2/b/Adam_7BCNN_encoder_cat/catout/WBCNN_encoder_cat/catout/W/AdamBCNN_encoder_cat/catout/W/Adam_1BCNN_encoder_cat/catout/W/Adam_2BCNN_encoder_cat/catout/W/Adam_3BCNN_encoder_cat/catout/W/Adam_4BCNN_encoder_cat/catout/W/Adam_5BCNN_encoder_cat/catout/W/Adam_6BCNN_encoder_cat/catout/W/Adam_7BCNN_encoder_cat/catout/bBCNN_encoder_cat/catout/b/AdamBCNN_encoder_cat/catout/b/Adam_1BCNN_encoder_cat/catout/b/Adam_2BCNN_encoder_cat/catout/b/Adam_3BCNN_encoder_cat/catout/b/Adam_4BCNN_encoder_cat/catout/b/Adam_5BCNN_encoder_cat/catout/b/Adam_6BCNN_encoder_cat/catout/b/Adam_7BCNN_encoder_cat/zout/WBCNN_encoder_cat/zout/W/AdamBCNN_encoder_cat/zout/W/Adam_1BCNN_encoder_cat/zout/W/Adam_2BCNN_encoder_cat/zout/W/Adam_3BCNN_encoder_cat/zout/W/Adam_4BCNN_encoder_cat/zout/W/Adam_5BCNN_encoder_cat/zout/bBCNN_encoder_cat/zout/b/AdamBCNN_encoder_cat/zout/b/Adam_1BCNN_encoder_cat/zout/b/Adam_2BCNN_encoder_cat/zout/b/Adam_3BCNN_encoder_cat/zout/b/Adam_4BCNN_encoder_cat/zout/b/Adam_5Bbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta1_power_3Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bbeta2_power_3Bdiscriminator/b0Bdiscriminator/b0/AdamBdiscriminator/b0/Adam_1Bdiscriminator/b1Bdiscriminator/b1/AdamBdiscriminator/b1/Adam_1Bdiscriminator/boBdiscriminator/bo/AdamBdiscriminator/bo/Adam_1Bdiscriminator/w0Bdiscriminator/w0/AdamBdiscriminator/w0/Adam_1Bdiscriminator/w1Bdiscriminator/w1/AdamBdiscriminator/w1/Adam_1Bdiscriminator/woBdiscriminator/wo/AdamBdiscriminator/wo/Adam_1Bdiscriminator_cat/b0Bdiscriminator_cat/b0/AdamBdiscriminator_cat/b0/Adam_1Bdiscriminator_cat/b1Bdiscriminator_cat/b1/AdamBdiscriminator_cat/b1/Adam_1Bdiscriminator_cat/boBdiscriminator_cat/bo/AdamBdiscriminator_cat/bo/Adam_1Bdiscriminator_cat/w0Bdiscriminator_cat/w0/AdamBdiscriminator_cat/w0/Adam_1Bdiscriminator_cat/w1Bdiscriminator_cat/w1/AdamBdiscriminator_cat/w1/Adam_1Bdiscriminator_cat/woBdiscriminator_cat/wo/AdamBdiscriminator_cat/wo/Adam_1Bis_training*
dtype0
«
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*ä
valueŚB×§B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
Æ
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*ø
dtypes­
Ŗ2§


save/AssignAssignCNN_decoder/Conv2D/Wsave/RestoreV2*
use_locking(*
T0*'
_class
loc:@CNN_decoder/Conv2D/W*
validate_shape(

save/Assign_1AssignCNN_decoder/Conv2D/W/Adamsave/RestoreV2:1*
use_locking(*
T0*'
_class
loc:@CNN_decoder/Conv2D/W*
validate_shape(
”
save/Assign_2AssignCNN_decoder/Conv2D/W/Adam_1save/RestoreV2:2*
use_locking(*
T0*'
_class
loc:@CNN_decoder/Conv2D/W*
validate_shape(

save/Assign_3AssignCNN_decoder/Conv2D/bsave/RestoreV2:3*
use_locking(*
T0*'
_class
loc:@CNN_decoder/Conv2D/b*
validate_shape(

save/Assign_4AssignCNN_decoder/Conv2D/b/Adamsave/RestoreV2:4*
use_locking(*
T0*'
_class
loc:@CNN_decoder/Conv2D/b*
validate_shape(
”
save/Assign_5AssignCNN_decoder/Conv2D/b/Adam_1save/RestoreV2:5*
use_locking(*
T0*'
_class
loc:@CNN_decoder/Conv2D/b*
validate_shape(

save/Assign_6AssignCNN_decoder/Conv2D_1/Wsave/RestoreV2:6*
use_locking(*
T0*)
_class
loc:@CNN_decoder/Conv2D_1/W*
validate_shape(
£
save/Assign_7AssignCNN_decoder/Conv2D_1/W/Adamsave/RestoreV2:7*
use_locking(*
T0*)
_class
loc:@CNN_decoder/Conv2D_1/W*
validate_shape(
„
save/Assign_8AssignCNN_decoder/Conv2D_1/W/Adam_1save/RestoreV2:8*
use_locking(*
T0*)
_class
loc:@CNN_decoder/Conv2D_1/W*
validate_shape(

save/Assign_9AssignCNN_decoder/Conv2D_1/bsave/RestoreV2:9*
use_locking(*
T0*)
_class
loc:@CNN_decoder/Conv2D_1/b*
validate_shape(
„
save/Assign_10AssignCNN_decoder/Conv2D_1/b/Adamsave/RestoreV2:10*
use_locking(*
T0*)
_class
loc:@CNN_decoder/Conv2D_1/b*
validate_shape(
§
save/Assign_11AssignCNN_decoder/Conv2D_1/b/Adam_1save/RestoreV2:11*
use_locking(*
T0*)
_class
loc:@CNN_decoder/Conv2D_1/b*
validate_shape(
 
save/Assign_12AssignCNN_decoder/Conv2D_2/Wsave/RestoreV2:12*
use_locking(*
T0*)
_class
loc:@CNN_decoder/Conv2D_2/W*
validate_shape(
„
save/Assign_13AssignCNN_decoder/Conv2D_2/W/Adamsave/RestoreV2:13*
use_locking(*
T0*)
_class
loc:@CNN_decoder/Conv2D_2/W*
validate_shape(
§
save/Assign_14AssignCNN_decoder/Conv2D_2/W/Adam_1save/RestoreV2:14*
use_locking(*
T0*)
_class
loc:@CNN_decoder/Conv2D_2/W*
validate_shape(
 
save/Assign_15AssignCNN_decoder/Conv2D_2/bsave/RestoreV2:15*
use_locking(*
T0*)
_class
loc:@CNN_decoder/Conv2D_2/b*
validate_shape(
„
save/Assign_16AssignCNN_decoder/Conv2D_2/b/Adamsave/RestoreV2:16*
use_locking(*
T0*)
_class
loc:@CNN_decoder/Conv2D_2/b*
validate_shape(
§
save/Assign_17AssignCNN_decoder/Conv2D_2/b/Adam_1save/RestoreV2:17*
use_locking(*
T0*)
_class
loc:@CNN_decoder/Conv2D_2/b*
validate_shape(
¬
save/Assign_18AssignCNN_decoder/FullyConnected/Wsave/RestoreV2:18*
use_locking(*
T0*/
_class%
#!loc:@CNN_decoder/FullyConnected/W*
validate_shape(
±
save/Assign_19Assign!CNN_decoder/FullyConnected/W/Adamsave/RestoreV2:19*
use_locking(*
T0*/
_class%
#!loc:@CNN_decoder/FullyConnected/W*
validate_shape(
³
save/Assign_20Assign#CNN_decoder/FullyConnected/W/Adam_1save/RestoreV2:20*
use_locking(*
T0*/
_class%
#!loc:@CNN_decoder/FullyConnected/W*
validate_shape(
¬
save/Assign_21AssignCNN_decoder/FullyConnected/bsave/RestoreV2:21*
use_locking(*
T0*/
_class%
#!loc:@CNN_decoder/FullyConnected/b*
validate_shape(
±
save/Assign_22Assign!CNN_decoder/FullyConnected/b/Adamsave/RestoreV2:22*
use_locking(*
T0*/
_class%
#!loc:@CNN_decoder/FullyConnected/b*
validate_shape(
³
save/Assign_23Assign#CNN_decoder/FullyConnected/b/Adam_1save/RestoreV2:23*
use_locking(*
T0*/
_class%
#!loc:@CNN_decoder/FullyConnected/b*
validate_shape(

save/Assign_24AssignCNN_decoder/sigout/Wsave/RestoreV2:24*
use_locking(*
T0*'
_class
loc:@CNN_decoder/sigout/W*
validate_shape(
”
save/Assign_25AssignCNN_decoder/sigout/W/Adamsave/RestoreV2:25*
use_locking(*
T0*'
_class
loc:@CNN_decoder/sigout/W*
validate_shape(
£
save/Assign_26AssignCNN_decoder/sigout/W/Adam_1save/RestoreV2:26*
use_locking(*
T0*'
_class
loc:@CNN_decoder/sigout/W*
validate_shape(

save/Assign_27AssignCNN_decoder/sigout/bsave/RestoreV2:27*
use_locking(*
T0*'
_class
loc:@CNN_decoder/sigout/b*
validate_shape(
”
save/Assign_28AssignCNN_decoder/sigout/b/Adamsave/RestoreV2:28*
use_locking(*
T0*'
_class
loc:@CNN_decoder/sigout/b*
validate_shape(
£
save/Assign_29AssignCNN_decoder/sigout/b/Adam_1save/RestoreV2:29*
use_locking(*
T0*'
_class
loc:@CNN_decoder/sigout/b*
validate_shape(
 
save/Assign_30AssignCNN_decoder/sigout_1/Wsave/RestoreV2:30*
use_locking(*
T0*)
_class
loc:@CNN_decoder/sigout_1/W*
validate_shape(
„
save/Assign_31AssignCNN_decoder/sigout_1/W/Adamsave/RestoreV2:31*
use_locking(*
T0*)
_class
loc:@CNN_decoder/sigout_1/W*
validate_shape(
§
save/Assign_32AssignCNN_decoder/sigout_1/W/Adam_1save/RestoreV2:32*
use_locking(*
T0*)
_class
loc:@CNN_decoder/sigout_1/W*
validate_shape(
 
save/Assign_33AssignCNN_decoder/sigout_1/bsave/RestoreV2:33*
use_locking(*
T0*)
_class
loc:@CNN_decoder/sigout_1/b*
validate_shape(
„
save/Assign_34AssignCNN_decoder/sigout_1/b/Adamsave/RestoreV2:34*
use_locking(*
T0*)
_class
loc:@CNN_decoder/sigout_1/b*
validate_shape(
§
save/Assign_35AssignCNN_decoder/sigout_1/b/Adam_1save/RestoreV2:35*
use_locking(*
T0*)
_class
loc:@CNN_decoder/sigout_1/b*
validate_shape(
¤
save/Assign_36AssignCNN_encoder_cat/Conv2D/Wsave/RestoreV2:36*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(
©
save/Assign_37AssignCNN_encoder_cat/Conv2D/W/Adamsave/RestoreV2:37*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(
«
save/Assign_38AssignCNN_encoder_cat/Conv2D/W/Adam_1save/RestoreV2:38*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(
«
save/Assign_39AssignCNN_encoder_cat/Conv2D/W/Adam_2save/RestoreV2:39*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(
«
save/Assign_40AssignCNN_encoder_cat/Conv2D/W/Adam_3save/RestoreV2:40*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(
«
save/Assign_41AssignCNN_encoder_cat/Conv2D/W/Adam_4save/RestoreV2:41*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(
«
save/Assign_42AssignCNN_encoder_cat/Conv2D/W/Adam_5save/RestoreV2:42*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(
«
save/Assign_43AssignCNN_encoder_cat/Conv2D/W/Adam_6save/RestoreV2:43*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(
«
save/Assign_44AssignCNN_encoder_cat/Conv2D/W/Adam_7save/RestoreV2:44*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(
¤
save/Assign_45AssignCNN_encoder_cat/Conv2D/bsave/RestoreV2:45*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
validate_shape(
©
save/Assign_46AssignCNN_encoder_cat/Conv2D/b/Adamsave/RestoreV2:46*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
validate_shape(
«
save/Assign_47AssignCNN_encoder_cat/Conv2D/b/Adam_1save/RestoreV2:47*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
validate_shape(
«
save/Assign_48AssignCNN_encoder_cat/Conv2D/b/Adam_2save/RestoreV2:48*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
validate_shape(
«
save/Assign_49AssignCNN_encoder_cat/Conv2D/b/Adam_3save/RestoreV2:49*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
validate_shape(
«
save/Assign_50AssignCNN_encoder_cat/Conv2D/b/Adam_4save/RestoreV2:50*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
validate_shape(
«
save/Assign_51AssignCNN_encoder_cat/Conv2D/b/Adam_5save/RestoreV2:51*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
validate_shape(
«
save/Assign_52AssignCNN_encoder_cat/Conv2D/b/Adam_6save/RestoreV2:52*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
validate_shape(
«
save/Assign_53AssignCNN_encoder_cat/Conv2D/b/Adam_7save/RestoreV2:53*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/b*
validate_shape(
Ø
save/Assign_54AssignCNN_encoder_cat/Conv2D_1/Wsave/RestoreV2:54*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
validate_shape(
­
save/Assign_55AssignCNN_encoder_cat/Conv2D_1/W/Adamsave/RestoreV2:55*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
validate_shape(
Æ
save/Assign_56Assign!CNN_encoder_cat/Conv2D_1/W/Adam_1save/RestoreV2:56*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
validate_shape(
Æ
save/Assign_57Assign!CNN_encoder_cat/Conv2D_1/W/Adam_2save/RestoreV2:57*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
validate_shape(
Æ
save/Assign_58Assign!CNN_encoder_cat/Conv2D_1/W/Adam_3save/RestoreV2:58*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
validate_shape(
Æ
save/Assign_59Assign!CNN_encoder_cat/Conv2D_1/W/Adam_4save/RestoreV2:59*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
validate_shape(
Æ
save/Assign_60Assign!CNN_encoder_cat/Conv2D_1/W/Adam_5save/RestoreV2:60*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
validate_shape(
Æ
save/Assign_61Assign!CNN_encoder_cat/Conv2D_1/W/Adam_6save/RestoreV2:61*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
validate_shape(
Æ
save/Assign_62Assign!CNN_encoder_cat/Conv2D_1/W/Adam_7save/RestoreV2:62*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/W*
validate_shape(
Ø
save/Assign_63AssignCNN_encoder_cat/Conv2D_1/bsave/RestoreV2:63*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
validate_shape(
­
save/Assign_64AssignCNN_encoder_cat/Conv2D_1/b/Adamsave/RestoreV2:64*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
validate_shape(
Æ
save/Assign_65Assign!CNN_encoder_cat/Conv2D_1/b/Adam_1save/RestoreV2:65*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
validate_shape(
Æ
save/Assign_66Assign!CNN_encoder_cat/Conv2D_1/b/Adam_2save/RestoreV2:66*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
validate_shape(
Æ
save/Assign_67Assign!CNN_encoder_cat/Conv2D_1/b/Adam_3save/RestoreV2:67*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
validate_shape(
Æ
save/Assign_68Assign!CNN_encoder_cat/Conv2D_1/b/Adam_4save/RestoreV2:68*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
validate_shape(
Æ
save/Assign_69Assign!CNN_encoder_cat/Conv2D_1/b/Adam_5save/RestoreV2:69*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
validate_shape(
Æ
save/Assign_70Assign!CNN_encoder_cat/Conv2D_1/b/Adam_6save/RestoreV2:70*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
validate_shape(
Æ
save/Assign_71Assign!CNN_encoder_cat/Conv2D_1/b/Adam_7save/RestoreV2:71*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_1/b*
validate_shape(
Ø
save/Assign_72AssignCNN_encoder_cat/Conv2D_2/Wsave/RestoreV2:72*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
validate_shape(
­
save/Assign_73AssignCNN_encoder_cat/Conv2D_2/W/Adamsave/RestoreV2:73*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
validate_shape(
Æ
save/Assign_74Assign!CNN_encoder_cat/Conv2D_2/W/Adam_1save/RestoreV2:74*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
validate_shape(
Æ
save/Assign_75Assign!CNN_encoder_cat/Conv2D_2/W/Adam_2save/RestoreV2:75*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
validate_shape(
Æ
save/Assign_76Assign!CNN_encoder_cat/Conv2D_2/W/Adam_3save/RestoreV2:76*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
validate_shape(
Æ
save/Assign_77Assign!CNN_encoder_cat/Conv2D_2/W/Adam_4save/RestoreV2:77*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
validate_shape(
Æ
save/Assign_78Assign!CNN_encoder_cat/Conv2D_2/W/Adam_5save/RestoreV2:78*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
validate_shape(
Æ
save/Assign_79Assign!CNN_encoder_cat/Conv2D_2/W/Adam_6save/RestoreV2:79*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
validate_shape(
Æ
save/Assign_80Assign!CNN_encoder_cat/Conv2D_2/W/Adam_7save/RestoreV2:80*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/W*
validate_shape(
Ø
save/Assign_81AssignCNN_encoder_cat/Conv2D_2/bsave/RestoreV2:81*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
validate_shape(
­
save/Assign_82AssignCNN_encoder_cat/Conv2D_2/b/Adamsave/RestoreV2:82*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
validate_shape(
Æ
save/Assign_83Assign!CNN_encoder_cat/Conv2D_2/b/Adam_1save/RestoreV2:83*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
validate_shape(
Æ
save/Assign_84Assign!CNN_encoder_cat/Conv2D_2/b/Adam_2save/RestoreV2:84*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
validate_shape(
Æ
save/Assign_85Assign!CNN_encoder_cat/Conv2D_2/b/Adam_3save/RestoreV2:85*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
validate_shape(
Æ
save/Assign_86Assign!CNN_encoder_cat/Conv2D_2/b/Adam_4save/RestoreV2:86*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
validate_shape(
Æ
save/Assign_87Assign!CNN_encoder_cat/Conv2D_2/b/Adam_5save/RestoreV2:87*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
validate_shape(
Æ
save/Assign_88Assign!CNN_encoder_cat/Conv2D_2/b/Adam_6save/RestoreV2:88*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
validate_shape(
Æ
save/Assign_89Assign!CNN_encoder_cat/Conv2D_2/b/Adam_7save/RestoreV2:89*
use_locking(*
T0*-
_class#
!loc:@CNN_encoder_cat/Conv2D_2/b*
validate_shape(
¤
save/Assign_90AssignCNN_encoder_cat/catout/Wsave/RestoreV2:90*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W*
validate_shape(
©
save/Assign_91AssignCNN_encoder_cat/catout/W/Adamsave/RestoreV2:91*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W*
validate_shape(
«
save/Assign_92AssignCNN_encoder_cat/catout/W/Adam_1save/RestoreV2:92*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W*
validate_shape(
«
save/Assign_93AssignCNN_encoder_cat/catout/W/Adam_2save/RestoreV2:93*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W*
validate_shape(
«
save/Assign_94AssignCNN_encoder_cat/catout/W/Adam_3save/RestoreV2:94*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W*
validate_shape(
«
save/Assign_95AssignCNN_encoder_cat/catout/W/Adam_4save/RestoreV2:95*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W*
validate_shape(
«
save/Assign_96AssignCNN_encoder_cat/catout/W/Adam_5save/RestoreV2:96*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W*
validate_shape(
«
save/Assign_97AssignCNN_encoder_cat/catout/W/Adam_6save/RestoreV2:97*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W*
validate_shape(
«
save/Assign_98AssignCNN_encoder_cat/catout/W/Adam_7save/RestoreV2:98*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/W*
validate_shape(
¤
save/Assign_99AssignCNN_encoder_cat/catout/bsave/RestoreV2:99*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b*
validate_shape(
«
save/Assign_100AssignCNN_encoder_cat/catout/b/Adamsave/RestoreV2:100*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b*
validate_shape(
­
save/Assign_101AssignCNN_encoder_cat/catout/b/Adam_1save/RestoreV2:101*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b*
validate_shape(
­
save/Assign_102AssignCNN_encoder_cat/catout/b/Adam_2save/RestoreV2:102*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b*
validate_shape(
­
save/Assign_103AssignCNN_encoder_cat/catout/b/Adam_3save/RestoreV2:103*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b*
validate_shape(
­
save/Assign_104AssignCNN_encoder_cat/catout/b/Adam_4save/RestoreV2:104*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b*
validate_shape(
­
save/Assign_105AssignCNN_encoder_cat/catout/b/Adam_5save/RestoreV2:105*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b*
validate_shape(
­
save/Assign_106AssignCNN_encoder_cat/catout/b/Adam_6save/RestoreV2:106*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b*
validate_shape(
­
save/Assign_107AssignCNN_encoder_cat/catout/b/Adam_7save/RestoreV2:107*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/catout/b*
validate_shape(
¢
save/Assign_108AssignCNN_encoder_cat/zout/Wsave/RestoreV2:108*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/W*
validate_shape(
§
save/Assign_109AssignCNN_encoder_cat/zout/W/Adamsave/RestoreV2:109*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/W*
validate_shape(
©
save/Assign_110AssignCNN_encoder_cat/zout/W/Adam_1save/RestoreV2:110*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/W*
validate_shape(
©
save/Assign_111AssignCNN_encoder_cat/zout/W/Adam_2save/RestoreV2:111*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/W*
validate_shape(
©
save/Assign_112AssignCNN_encoder_cat/zout/W/Adam_3save/RestoreV2:112*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/W*
validate_shape(
©
save/Assign_113AssignCNN_encoder_cat/zout/W/Adam_4save/RestoreV2:113*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/W*
validate_shape(
©
save/Assign_114AssignCNN_encoder_cat/zout/W/Adam_5save/RestoreV2:114*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/W*
validate_shape(
¢
save/Assign_115AssignCNN_encoder_cat/zout/bsave/RestoreV2:115*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/b*
validate_shape(
§
save/Assign_116AssignCNN_encoder_cat/zout/b/Adamsave/RestoreV2:116*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/b*
validate_shape(
©
save/Assign_117AssignCNN_encoder_cat/zout/b/Adam_1save/RestoreV2:117*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/b*
validate_shape(
©
save/Assign_118AssignCNN_encoder_cat/zout/b/Adam_2save/RestoreV2:118*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/b*
validate_shape(
©
save/Assign_119AssignCNN_encoder_cat/zout/b/Adam_3save/RestoreV2:119*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/b*
validate_shape(
©
save/Assign_120AssignCNN_encoder_cat/zout/b/Adam_4save/RestoreV2:120*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/b*
validate_shape(
©
save/Assign_121AssignCNN_encoder_cat/zout/b/Adam_5save/RestoreV2:121*
use_locking(*
T0*)
_class
loc:@CNN_encoder_cat/zout/b*
validate_shape(

save/Assign_122Assignbeta1_powersave/RestoreV2:122*
use_locking(*
T0*'
_class
loc:@CNN_decoder/Conv2D/W*
validate_shape(

save/Assign_123Assignbeta1_power_1save/RestoreV2:123*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(

save/Assign_124Assignbeta1_power_2save/RestoreV2:124*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(

save/Assign_125Assignbeta1_power_3save/RestoreV2:125*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(

save/Assign_126Assignbeta2_powersave/RestoreV2:126*
use_locking(*
T0*'
_class
loc:@CNN_decoder/Conv2D/W*
validate_shape(

save/Assign_127Assignbeta2_power_1save/RestoreV2:127*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(

save/Assign_128Assignbeta2_power_2save/RestoreV2:128*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(

save/Assign_129Assignbeta2_power_3save/RestoreV2:129*
use_locking(*
T0*+
_class!
loc:@CNN_encoder_cat/Conv2D/W*
validate_shape(

save/Assign_130Assigndiscriminator/b0save/RestoreV2:130*
use_locking(*
T0*#
_class
loc:@discriminator/b0*
validate_shape(

save/Assign_131Assigndiscriminator/b0/Adamsave/RestoreV2:131*
use_locking(*
T0*#
_class
loc:@discriminator/b0*
validate_shape(

save/Assign_132Assigndiscriminator/b0/Adam_1save/RestoreV2:132*
use_locking(*
T0*#
_class
loc:@discriminator/b0*
validate_shape(

save/Assign_133Assigndiscriminator/b1save/RestoreV2:133*
use_locking(*
T0*#
_class
loc:@discriminator/b1*
validate_shape(

save/Assign_134Assigndiscriminator/b1/Adamsave/RestoreV2:134*
use_locking(*
T0*#
_class
loc:@discriminator/b1*
validate_shape(

save/Assign_135Assigndiscriminator/b1/Adam_1save/RestoreV2:135*
use_locking(*
T0*#
_class
loc:@discriminator/b1*
validate_shape(

save/Assign_136Assigndiscriminator/bosave/RestoreV2:136*
use_locking(*
T0*#
_class
loc:@discriminator/bo*
validate_shape(

save/Assign_137Assigndiscriminator/bo/Adamsave/RestoreV2:137*
use_locking(*
T0*#
_class
loc:@discriminator/bo*
validate_shape(

save/Assign_138Assigndiscriminator/bo/Adam_1save/RestoreV2:138*
use_locking(*
T0*#
_class
loc:@discriminator/bo*
validate_shape(

save/Assign_139Assigndiscriminator/w0save/RestoreV2:139*
use_locking(*
T0*#
_class
loc:@discriminator/w0*
validate_shape(

save/Assign_140Assigndiscriminator/w0/Adamsave/RestoreV2:140*
use_locking(*
T0*#
_class
loc:@discriminator/w0*
validate_shape(

save/Assign_141Assigndiscriminator/w0/Adam_1save/RestoreV2:141*
use_locking(*
T0*#
_class
loc:@discriminator/w0*
validate_shape(

save/Assign_142Assigndiscriminator/w1save/RestoreV2:142*
use_locking(*
T0*#
_class
loc:@discriminator/w1*
validate_shape(

save/Assign_143Assigndiscriminator/w1/Adamsave/RestoreV2:143*
use_locking(*
T0*#
_class
loc:@discriminator/w1*
validate_shape(

save/Assign_144Assigndiscriminator/w1/Adam_1save/RestoreV2:144*
use_locking(*
T0*#
_class
loc:@discriminator/w1*
validate_shape(

save/Assign_145Assigndiscriminator/wosave/RestoreV2:145*
use_locking(*
T0*#
_class
loc:@discriminator/wo*
validate_shape(

save/Assign_146Assigndiscriminator/wo/Adamsave/RestoreV2:146*
use_locking(*
T0*#
_class
loc:@discriminator/wo*
validate_shape(

save/Assign_147Assigndiscriminator/wo/Adam_1save/RestoreV2:147*
use_locking(*
T0*#
_class
loc:@discriminator/wo*
validate_shape(

save/Assign_148Assigndiscriminator_cat/b0save/RestoreV2:148*
use_locking(*
T0*'
_class
loc:@discriminator_cat/b0*
validate_shape(
£
save/Assign_149Assigndiscriminator_cat/b0/Adamsave/RestoreV2:149*
use_locking(*
T0*'
_class
loc:@discriminator_cat/b0*
validate_shape(
„
save/Assign_150Assigndiscriminator_cat/b0/Adam_1save/RestoreV2:150*
use_locking(*
T0*'
_class
loc:@discriminator_cat/b0*
validate_shape(

save/Assign_151Assigndiscriminator_cat/b1save/RestoreV2:151*
use_locking(*
T0*'
_class
loc:@discriminator_cat/b1*
validate_shape(
£
save/Assign_152Assigndiscriminator_cat/b1/Adamsave/RestoreV2:152*
use_locking(*
T0*'
_class
loc:@discriminator_cat/b1*
validate_shape(
„
save/Assign_153Assigndiscriminator_cat/b1/Adam_1save/RestoreV2:153*
use_locking(*
T0*'
_class
loc:@discriminator_cat/b1*
validate_shape(

save/Assign_154Assigndiscriminator_cat/bosave/RestoreV2:154*
use_locking(*
T0*'
_class
loc:@discriminator_cat/bo*
validate_shape(
£
save/Assign_155Assigndiscriminator_cat/bo/Adamsave/RestoreV2:155*
use_locking(*
T0*'
_class
loc:@discriminator_cat/bo*
validate_shape(
„
save/Assign_156Assigndiscriminator_cat/bo/Adam_1save/RestoreV2:156*
use_locking(*
T0*'
_class
loc:@discriminator_cat/bo*
validate_shape(

save/Assign_157Assigndiscriminator_cat/w0save/RestoreV2:157*
use_locking(*
T0*'
_class
loc:@discriminator_cat/w0*
validate_shape(
£
save/Assign_158Assigndiscriminator_cat/w0/Adamsave/RestoreV2:158*
use_locking(*
T0*'
_class
loc:@discriminator_cat/w0*
validate_shape(
„
save/Assign_159Assigndiscriminator_cat/w0/Adam_1save/RestoreV2:159*
use_locking(*
T0*'
_class
loc:@discriminator_cat/w0*
validate_shape(

save/Assign_160Assigndiscriminator_cat/w1save/RestoreV2:160*
use_locking(*
T0*'
_class
loc:@discriminator_cat/w1*
validate_shape(
£
save/Assign_161Assigndiscriminator_cat/w1/Adamsave/RestoreV2:161*
use_locking(*
T0*'
_class
loc:@discriminator_cat/w1*
validate_shape(
„
save/Assign_162Assigndiscriminator_cat/w1/Adam_1save/RestoreV2:162*
use_locking(*
T0*'
_class
loc:@discriminator_cat/w1*
validate_shape(

save/Assign_163Assigndiscriminator_cat/wosave/RestoreV2:163*
use_locking(*
T0*'
_class
loc:@discriminator_cat/wo*
validate_shape(
£
save/Assign_164Assigndiscriminator_cat/wo/Adamsave/RestoreV2:164*
use_locking(*
T0*'
_class
loc:@discriminator_cat/wo*
validate_shape(
„
save/Assign_165Assigndiscriminator_cat/wo/Adam_1save/RestoreV2:165*
use_locking(*
T0*'
_class
loc:@discriminator_cat/wo*
validate_shape(

save/Assign_166Assignis_trainingsave/RestoreV2:166*
use_locking(*
T0
*
_class
loc:@is_training*
validate_shape(
ę
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_100^save/Assign_101^save/Assign_102^save/Assign_103^save/Assign_104^save/Assign_105^save/Assign_106^save/Assign_107^save/Assign_108^save/Assign_109^save/Assign_11^save/Assign_110^save/Assign_111^save/Assign_112^save/Assign_113^save/Assign_114^save/Assign_115^save/Assign_116^save/Assign_117^save/Assign_118^save/Assign_119^save/Assign_12^save/Assign_120^save/Assign_121^save/Assign_122^save/Assign_123^save/Assign_124^save/Assign_125^save/Assign_126^save/Assign_127^save/Assign_128^save/Assign_129^save/Assign_13^save/Assign_130^save/Assign_131^save/Assign_132^save/Assign_133^save/Assign_134^save/Assign_135^save/Assign_136^save/Assign_137^save/Assign_138^save/Assign_139^save/Assign_14^save/Assign_140^save/Assign_141^save/Assign_142^save/Assign_143^save/Assign_144^save/Assign_145^save/Assign_146^save/Assign_147^save/Assign_148^save/Assign_149^save/Assign_15^save/Assign_150^save/Assign_151^save/Assign_152^save/Assign_153^save/Assign_154^save/Assign_155^save/Assign_156^save/Assign_157^save/Assign_158^save/Assign_159^save/Assign_16^save/Assign_160^save/Assign_161^save/Assign_162^save/Assign_163^save/Assign_164^save/Assign_165^save/Assign_166^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_7^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75^save/Assign_76^save/Assign_77^save/Assign_78^save/Assign_79^save/Assign_8^save/Assign_80^save/Assign_81^save/Assign_82^save/Assign_83^save/Assign_84^save/Assign_85^save/Assign_86^save/Assign_87^save/Assign_88^save/Assign_89^save/Assign_9^save/Assign_90^save/Assign_91^save/Assign_92^save/Assign_93^save/Assign_94^save/Assign_95^save/Assign_96^save/Assign_97^save/Assign_98^save/Assign_99
0
initNoOp!^CNN_decoder/Conv2D/W/Adam/Assign#^CNN_decoder/Conv2D/W/Adam_1/Assign^CNN_decoder/Conv2D/W/Assign!^CNN_decoder/Conv2D/b/Adam/Assign#^CNN_decoder/Conv2D/b/Adam_1/Assign^CNN_decoder/Conv2D/b/Assign#^CNN_decoder/Conv2D_1/W/Adam/Assign%^CNN_decoder/Conv2D_1/W/Adam_1/Assign^CNN_decoder/Conv2D_1/W/Assign#^CNN_decoder/Conv2D_1/b/Adam/Assign%^CNN_decoder/Conv2D_1/b/Adam_1/Assign^CNN_decoder/Conv2D_1/b/Assign#^CNN_decoder/Conv2D_2/W/Adam/Assign%^CNN_decoder/Conv2D_2/W/Adam_1/Assign^CNN_decoder/Conv2D_2/W/Assign#^CNN_decoder/Conv2D_2/b/Adam/Assign%^CNN_decoder/Conv2D_2/b/Adam_1/Assign^CNN_decoder/Conv2D_2/b/Assign)^CNN_decoder/FullyConnected/W/Adam/Assign+^CNN_decoder/FullyConnected/W/Adam_1/Assign$^CNN_decoder/FullyConnected/W/Assign)^CNN_decoder/FullyConnected/b/Adam/Assign+^CNN_decoder/FullyConnected/b/Adam_1/Assign$^CNN_decoder/FullyConnected/b/Assign!^CNN_decoder/sigout/W/Adam/Assign#^CNN_decoder/sigout/W/Adam_1/Assign^CNN_decoder/sigout/W/Assign!^CNN_decoder/sigout/b/Adam/Assign#^CNN_decoder/sigout/b/Adam_1/Assign^CNN_decoder/sigout/b/Assign#^CNN_decoder/sigout_1/W/Adam/Assign%^CNN_decoder/sigout_1/W/Adam_1/Assign^CNN_decoder/sigout_1/W/Assign#^CNN_decoder/sigout_1/b/Adam/Assign%^CNN_decoder/sigout_1/b/Adam_1/Assign^CNN_decoder/sigout_1/b/Assign%^CNN_encoder_cat/Conv2D/W/Adam/Assign'^CNN_encoder_cat/Conv2D/W/Adam_1/Assign'^CNN_encoder_cat/Conv2D/W/Adam_2/Assign'^CNN_encoder_cat/Conv2D/W/Adam_3/Assign'^CNN_encoder_cat/Conv2D/W/Adam_4/Assign'^CNN_encoder_cat/Conv2D/W/Adam_5/Assign'^CNN_encoder_cat/Conv2D/W/Adam_6/Assign'^CNN_encoder_cat/Conv2D/W/Adam_7/Assign ^CNN_encoder_cat/Conv2D/W/Assign%^CNN_encoder_cat/Conv2D/b/Adam/Assign'^CNN_encoder_cat/Conv2D/b/Adam_1/Assign'^CNN_encoder_cat/Conv2D/b/Adam_2/Assign'^CNN_encoder_cat/Conv2D/b/Adam_3/Assign'^CNN_encoder_cat/Conv2D/b/Adam_4/Assign'^CNN_encoder_cat/Conv2D/b/Adam_5/Assign'^CNN_encoder_cat/Conv2D/b/Adam_6/Assign'^CNN_encoder_cat/Conv2D/b/Adam_7/Assign ^CNN_encoder_cat/Conv2D/b/Assign'^CNN_encoder_cat/Conv2D_1/W/Adam/Assign)^CNN_encoder_cat/Conv2D_1/W/Adam_1/Assign)^CNN_encoder_cat/Conv2D_1/W/Adam_2/Assign)^CNN_encoder_cat/Conv2D_1/W/Adam_3/Assign)^CNN_encoder_cat/Conv2D_1/W/Adam_4/Assign)^CNN_encoder_cat/Conv2D_1/W/Adam_5/Assign)^CNN_encoder_cat/Conv2D_1/W/Adam_6/Assign)^CNN_encoder_cat/Conv2D_1/W/Adam_7/Assign"^CNN_encoder_cat/Conv2D_1/W/Assign'^CNN_encoder_cat/Conv2D_1/b/Adam/Assign)^CNN_encoder_cat/Conv2D_1/b/Adam_1/Assign)^CNN_encoder_cat/Conv2D_1/b/Adam_2/Assign)^CNN_encoder_cat/Conv2D_1/b/Adam_3/Assign)^CNN_encoder_cat/Conv2D_1/b/Adam_4/Assign)^CNN_encoder_cat/Conv2D_1/b/Adam_5/Assign)^CNN_encoder_cat/Conv2D_1/b/Adam_6/Assign)^CNN_encoder_cat/Conv2D_1/b/Adam_7/Assign"^CNN_encoder_cat/Conv2D_1/b/Assign'^CNN_encoder_cat/Conv2D_2/W/Adam/Assign)^CNN_encoder_cat/Conv2D_2/W/Adam_1/Assign)^CNN_encoder_cat/Conv2D_2/W/Adam_2/Assign)^CNN_encoder_cat/Conv2D_2/W/Adam_3/Assign)^CNN_encoder_cat/Conv2D_2/W/Adam_4/Assign)^CNN_encoder_cat/Conv2D_2/W/Adam_5/Assign)^CNN_encoder_cat/Conv2D_2/W/Adam_6/Assign)^CNN_encoder_cat/Conv2D_2/W/Adam_7/Assign"^CNN_encoder_cat/Conv2D_2/W/Assign'^CNN_encoder_cat/Conv2D_2/b/Adam/Assign)^CNN_encoder_cat/Conv2D_2/b/Adam_1/Assign)^CNN_encoder_cat/Conv2D_2/b/Adam_2/Assign)^CNN_encoder_cat/Conv2D_2/b/Adam_3/Assign)^CNN_encoder_cat/Conv2D_2/b/Adam_4/Assign)^CNN_encoder_cat/Conv2D_2/b/Adam_5/Assign)^CNN_encoder_cat/Conv2D_2/b/Adam_6/Assign)^CNN_encoder_cat/Conv2D_2/b/Adam_7/Assign"^CNN_encoder_cat/Conv2D_2/b/Assign%^CNN_encoder_cat/catout/W/Adam/Assign'^CNN_encoder_cat/catout/W/Adam_1/Assign'^CNN_encoder_cat/catout/W/Adam_2/Assign'^CNN_encoder_cat/catout/W/Adam_3/Assign'^CNN_encoder_cat/catout/W/Adam_4/Assign'^CNN_encoder_cat/catout/W/Adam_5/Assign'^CNN_encoder_cat/catout/W/Adam_6/Assign'^CNN_encoder_cat/catout/W/Adam_7/Assign ^CNN_encoder_cat/catout/W/Assign%^CNN_encoder_cat/catout/b/Adam/Assign'^CNN_encoder_cat/catout/b/Adam_1/Assign'^CNN_encoder_cat/catout/b/Adam_2/Assign'^CNN_encoder_cat/catout/b/Adam_3/Assign'^CNN_encoder_cat/catout/b/Adam_4/Assign'^CNN_encoder_cat/catout/b/Adam_5/Assign'^CNN_encoder_cat/catout/b/Adam_6/Assign'^CNN_encoder_cat/catout/b/Adam_7/Assign ^CNN_encoder_cat/catout/b/Assign#^CNN_encoder_cat/zout/W/Adam/Assign%^CNN_encoder_cat/zout/W/Adam_1/Assign%^CNN_encoder_cat/zout/W/Adam_2/Assign%^CNN_encoder_cat/zout/W/Adam_3/Assign%^CNN_encoder_cat/zout/W/Adam_4/Assign%^CNN_encoder_cat/zout/W/Adam_5/Assign^CNN_encoder_cat/zout/W/Assign#^CNN_encoder_cat/zout/b/Adam/Assign%^CNN_encoder_cat/zout/b/Adam_1/Assign%^CNN_encoder_cat/zout/b/Adam_2/Assign%^CNN_encoder_cat/zout/b/Adam_3/Assign%^CNN_encoder_cat/zout/b/Adam_4/Assign%^CNN_encoder_cat/zout/b/Adam_5/Assign^CNN_encoder_cat/zout/b/Assign^beta1_power/Assign^beta1_power_1/Assign^beta1_power_2/Assign^beta1_power_3/Assign^beta2_power/Assign^beta2_power_1/Assign^beta2_power_2/Assign^beta2_power_3/Assign^discriminator/b0/Adam/Assign^discriminator/b0/Adam_1/Assign^discriminator/b0/Assign^discriminator/b1/Adam/Assign^discriminator/b1/Adam_1/Assign^discriminator/b1/Assign^discriminator/bo/Adam/Assign^discriminator/bo/Adam_1/Assign^discriminator/bo/Assign^discriminator/w0/Adam/Assign^discriminator/w0/Adam_1/Assign^discriminator/w0/Assign^discriminator/w1/Adam/Assign^discriminator/w1/Adam_1/Assign^discriminator/w1/Assign^discriminator/wo/Adam/Assign^discriminator/wo/Adam_1/Assign^discriminator/wo/Assign!^discriminator_cat/b0/Adam/Assign#^discriminator_cat/b0/Adam_1/Assign^discriminator_cat/b0/Assign!^discriminator_cat/b1/Adam/Assign#^discriminator_cat/b1/Adam_1/Assign^discriminator_cat/b1/Assign!^discriminator_cat/bo/Adam/Assign#^discriminator_cat/bo/Adam_1/Assign^discriminator_cat/bo/Assign!^discriminator_cat/w0/Adam/Assign#^discriminator_cat/w0/Adam_1/Assign^discriminator_cat/w0/Assign!^discriminator_cat/w1/Adam/Assign#^discriminator_cat/w1/Adam_1/Assign^discriminator_cat/w1/Assign!^discriminator_cat/wo/Adam/Assign#^discriminator_cat/wo/Adam_1/Assign^discriminator_cat/wo/Assign^is_training/Assign"