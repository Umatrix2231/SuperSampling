using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow;
using static Tensorflow.Binding;
using Tensorflow.Keras;
using System.Collections.Concurrent;
using System.Threading.Tasks;

using System.Text;
using System.Threading;
using System.Windows.Forms;
using OpenCvSharp.Extensions;
using System.Security.Cryptography;
using Tensorflow.Keras.Engine;
using System.Threading.Tasks.Sources;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Gradients;
using System.Drawing;
using System.Runtime.InteropServices;
namespace SS1
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }
        public static string ReadFile()
        {
            OpenFileDialog openFileDialog = new OpenFileDialog();
            openFileDialog.InitialDirectory = System.Environment.CurrentDirectory;
            openFileDialog.Filter = "文件|*.bmp";

            openFileDialog.RestoreDirectory = true;
            openFileDialog.FilterIndex = 1;
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                return openFileDialog.FileName;
            }
            return null;
        }
        public static string SaveFile()
        {
            SaveFileDialog saveFileDialog = new SaveFileDialog();
            saveFileDialog.InitialDirectory = System.Environment.CurrentDirectory;
            saveFileDialog.Filter = "文件|*.bmp";

            saveFileDialog.RestoreDirectory = true;
            saveFileDialog.FilterIndex = 1;
            if (saveFileDialog.ShowDialog() == DialogResult.OK)
            {
                return saveFileDialog.FileName;
            }
            return null;
        }
        public static unsafe NDArray GetNDArray( string Filepath = null)
        {
            Mat Src = new Mat(Filepath);

            List<float[][][]> blist = new List<float[][][]>();
            float[][][] inputf = new float[Src.Height][][].Select(f => f = new float[Src.Width][].Select(k => k = new float[Src.Channels()]).ToArray()).ToArray();
            for (int y = 0; y < Src.Height; y++)
                for (int x = 0; x < Src.Width; x++)
                {
                    var srcptr = (byte*)Src.Ptr(y, x);
                    for (int i = 0; i < Src.Channels(); i++)
                    {
                        inputf[y][x][i] = srcptr[i];
                    }

                } 
            blist.Add(inputf);

            return np.array(blist.ToArray());
        }

        public static Tensor Padding(Tensor input, int Ysize, int Xsize)
        { 
            Tensor x = input;
            for (int i = 0; i < Ysize; i++)
            {
                Tensor r1 = tf.slice(x, new[] { 0, i * 2, 0, 0 }, new[] { -1, 1, -1, -1 });
                Tensor r2 = tf.slice(x, new[] { 0, x.shape[1] - 1 - i * 2, 0, 0 }, new[] { -1, 1, -1, -1 });
                x = tf.concat(new[] { r1, x, r2 }, 1);
            }
            for (int i = 0; i < Xsize; i++)
            {
                Tensor r11 = tf.slice(x, new[] { 0, 0, i * 2, 0 }, new[] { -1, -1, 1, -1 });
                Tensor r22 = tf.slice(x, new[] { 0, 0, x.shape[2] - 1 - i * 2, 0 }, new[] { -1, -1, 1, -1 });
                x = tf.concat(new[] { r11, x, r22 }, 2);
            }
            return x;
        }
        public static Tensor conv2d_st(Tensor x, IVariableV1 W, int stridesY, int stridesX, bool valid)
        {
            return tf.nn.conv2d(x, W, strides: new int[] { 1, stridesY, stridesX, 1 }, padding: valid ? "VALID" : "SAME");
        }
        public Tensor Conv2DCompute(Tensor input,ResourceVariable Weight, ResourceVariable Bias, int []Ksize)
        { 
            if (Ksize[0] > 2 && Ksize[1] == 1) input = Padding(input, Ksize[0] / 2, 0);
            if (Ksize[0] == 1 && Ksize[1] > 2) input = Padding(input, 0, Ksize[1] / 2);
            if (Ksize[0] != 1 && Ksize[1] != 1) input = Padding(input, Ksize[0] / 2, Ksize[1] / 2);
             
            input = conv2d_st(input, Weight, 1, 1, true) + Bias; 
            return input;
        }
        public static Tensor Resize1(Tensor x)
        {
            int shape3 = x.shape[3];
            int H = x.shape[1] * 2;
            int W = x.shape[2] * 2;

            Tensor[] input = tf.split(x, 4, 3);

            Tensor x1 = tf.concat(new Tensor[] { input[0], input[1] }, 3);
            Tensor x2 = tf.concat(new Tensor[] { input[2], input[3] }, 3);
            Tensor x3 = tf.concat(new Tensor[] { x1, x2 }, 2);
            x = tf.reshape(x3, (-1, H, W, shape3 / 4));

            return x;
        }
        public static Tensor Resize2(Tensor x)
        {
            int shape3 = x.shape[3];
            int H = x.shape[1] * 2;
            int W = x.shape[2] * 2;
             
            x = tf.concat(new Tensor[] { x, x }, 3);
            x = tf.concat(new Tensor[] { x, x }, 2);
            x = tf.reshape(x, (-1, H, W, shape3));
             
            return x; 
        }
        public ResourceVariable RetParams(string FilePath)
        {
            var x = np.load(FilePath);
            return tf.Variable(x, dtype: TF_DataType.TF_FLOAT, shape: x.shape);
        }
        private void Process (string FilePath)
        {
            var Sess = tf.Session(); 
            NDArray inputs = GetNDArray(FilePath);
            Tensor InputT = tf.constant(inputs, TF_DataType.TF_FLOAT, inputs.shape);

            var T0W = RetParams(@"model\TE1W_0.npy");
            var T1W = RetParams(@"model\TE1W_1.npy");
            var T2W = RetParams(@"model\TE1W_2.npy");
            var T3W = RetParams(@"model\TE1W_3.npy");
            var T4W = RetParams(@"model\TE1W_4.npy");
            var T5W = RetParams(@"model\TE1W_5.npy");
            var T6W = RetParams(@"model\TE1W_6.npy");

            var T0B = RetParams(@"model\TE1B_0.npy");
            var T1B = RetParams(@"model\TE1B_1.npy");
            var T2B = RetParams(@"model\TE1B_2.npy");
            var T3B = RetParams(@"model\TE1B_3.npy");
            var T4B = RetParams(@"model\TE1B_4.npy");
            var T5B = RetParams(@"model\TE1B_5.npy");
            var T6B = RetParams(@"model\TE1B_6.npy");

            var G0W = RetParams(@"model\TE2W_0.npy");
            var G1W = RetParams(@"model\TE2W_1.npy");
            var G2W = RetParams(@"model\TE2W_2.npy");
            var G3W = RetParams(@"model\TE2W_3.npy");
            var G4W = RetParams(@"model\TE2W_4.npy");
            var G5W = RetParams(@"model\TE2W_5.npy");
            var G6W = RetParams(@"model\TE2W_6.npy");

            var G0B = RetParams(@"model\TE2B_0.npy");
            var G1B = RetParams(@"model\TE2B_1.npy");
            var G2B = RetParams(@"model\TE2B_2.npy");
            var G3B = RetParams(@"model\TE2B_3.npy");
            var G4B = RetParams(@"model\TE2B_4.npy");
            var G5B = RetParams(@"model\TE2B_5.npy");
            var G6B = RetParams(@"model\TE2B_6.npy");


            InputT = (tf.nn.leaky_relu(Conv2DCompute(InputT, T0W, T0B, new[] { 7, 7 }), 0.1f));
            InputT = (tf.nn.leaky_relu(Conv2DCompute(InputT, T1W, T1B, new[] { 3, 3 }), 0.1f));
            InputT = (tf.nn.leaky_relu(Conv2DCompute(InputT, T2W, T2B, new[] { 3, 3 }), 0.1f));
            InputT = (tf.nn.leaky_relu(Conv2DCompute(InputT, T3W, T3B, new[] { 3, 3 }), 0.1f));
            InputT = (tf.nn.leaky_relu(Conv2DCompute(InputT, T4W, T4B, new[] { 3, 3 }), 0.1f));

            var InputG = (tf.nn.leaky_relu(Conv2DCompute(InputT, G0W, G0B, new[] { 3, 3 }), 0.1f));
            InputG = (tf.nn.leaky_relu(Conv2DCompute(InputG, G1W, G1B, new[] { 3, 3 }), 0.1f));
            InputG = (tf.nn.leaky_relu(Conv2DCompute(InputG, G2W, G2B, new[] { 3, 3 }), 0.1f));
            InputG = (tf.nn.leaky_relu(Conv2DCompute(InputG, G3W, G3B, new[] { 3, 3 }), 0.1f));
            InputG = (tf.nn.leaky_relu(Conv2DCompute(InputG, G4W, G4B, new[] { 3, 3 }), 0.1f));
            InputG = (tf.nn.leaky_relu(Conv2DCompute(InputG, G5W, G5B, new[] { 3, 3 }), 0.1f));
            InputG = (tf.nn.sigmoid(Conv2DCompute(InputG, G6W, G6B, new[] { 3, 3 })));

            InputT = (Resize2(InputT));
            InputG = (Resize1(InputG));

            InputT = (InputT * InputG);
            InputT = (tf.nn.leaky_relu(Conv2DCompute(InputT, T5W, T5B, new[] { 3, 3 }), 0.1f));
            InputT = (tf.nn.leaky_relu(Conv2DCompute(InputT, T6W, T6B, new[] { 7, 7 }), 0.1f));

            Sess.run(tf.global_variables_initializer());



            var Res=Sess.run(InputT)[0];


            var Arrays = (float[,,])(Res).ToMuliDimArray<float>();
            Mat mat = new Mat(Res.shape[0], Res.shape[1], MatType.CV_32FC3, Arrays);
            BeginInvoke(new Action(() => { mat.SaveImage(SaveFile()); }));
        }
        private void button1_Click(object sender, EventArgs e)
        {
            string FilePath = ReadFile();
            if (FilePath == null) return;
            new Thread(() => { Process(FilePath); }).Start();
            
        }
    }
}
