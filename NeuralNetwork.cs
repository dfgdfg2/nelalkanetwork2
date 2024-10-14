using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    public class NeuralNetwork
    {
        public List<Layer> neuronLayer { get; private set; }
        public Topology Topology { get; private set; }
        public NeuralNetwork(Topology topology)
        {
            this.Topology = topology;
            neuronLayer = new List<Layer>();

            InitializeNeurons();
        }

        public double Learn(List<double> inputs, int[] expected, int epoch, AbGradientDescent descent = null, Normalization normalization = Normalization.None)
        {
            if (descent == null)
            {
                descent = new GradientDescent(this);
            }

            return descent.Backpropagation(inputs, expected, epoch);
        }

        public double Learn(double[][] dataset, int[] expected, int epoch, AbGradientDescent descent = null, Normalization normalization = Normalization.None)
        {
            if (descent == null)
            {
                descent = new GradientDescent(this);
            }
            if (normalization == Normalization.NormalizationZ)
            {
                NormalizationZ(dataset);
            }
            else if (normalization == Normalization.MinMax)
            {
                MinMax(dataset);
            }

            double error = descent.Backpropagation(dataset, expected, epoch);
            
            return error;
        }

        public double FeedForward(List<double> inputs)
        {
            DefaultOptimizer toolNeuron = new DefaultOptimizer();
            Neuron currentNeuron = null;
            List<double> currentOutputs = FeedForwardInputs(toolNeuron, currentNeuron, inputs);

            for (int i = 1; i < neuronLayer.Count; i++)
            {
                double[] outputs = new double[neuronLayer[i].Count];
                for (int j = 0; j < neuronLayer[i].Count; j++)
                {
                    currentNeuron = neuronLayer[i][j];
                    double output = toolNeuron.FeedForward(currentOutputs, currentNeuron);
                    outputs[j] = output;
                }
                currentOutputs = new List<double>(outputs);
            }

            return currentNeuron.Output;
        }

        public double FeedForward(double[] inputs)
        {
            DefaultOptimizer toolNeuron = new DefaultOptimizer();
            Neuron currentNeuron = null;
            List<double> currentOutputs = FeedForwardInputs(toolNeuron, currentNeuron, inputs);

            for (int i = 1; i < neuronLayer.Count; i++)
            {
                double[] outputs = new double[neuronLayer[i].Count];
                Layer currentLayer = neuronLayer[i];
                for (int j = 0; j < neuronLayer[i].Count; j++)
                {
                    currentNeuron = currentLayer[j];
                    double output = toolNeuron.FeedForward(currentOutputs, currentNeuron);
                    outputs[j] = output;
                }
                currentOutputs = new List<double>(outputs);
            }

            return currentNeuron.Output;
        }

        public List<double> FeedForward(List<List<double>> inputs)
        {
            List<double> outputs = new List<double>();
            for (int i = 0; i < inputs.Count; i++)
            {
                outputs.Add(FeedForward(inputs[i]));
            }
            return outputs;
        }

        private List<double> FeedForwardInputs(DefaultOptimizer toolNeuron, Neuron currentNeuron, List<double> inputs)
        {
            List<double> outputs = new List<double>();
            for (int j = 0; j < neuronLayer[0].Count; j++)
            {
                currentNeuron = neuronLayer[0][j];
                List<double> inputs2 = new List<double> { inputs[j] };
                double output = toolNeuron.FeedForward(inputs2, currentNeuron);
                outputs.Add(output);
            }
            return outputs;
        }

        private List<double> FeedForwardInputs(DefaultOptimizer toolNeuron, Neuron currentNeuron, double[] inputs)
        {
            List<double> outputs = new List<double>();
            for (int j = 0; j < neuronLayer[0].Count; j++)
            {
                currentNeuron = neuronLayer[0][j];
                List<double> inputs2 = new List<double> { inputs[j] };
                double output = toolNeuron.FeedForward(inputs2, currentNeuron);
                outputs.Add(output);
            }
            return outputs;
        }

        private void NormalizationZ(double[][] inputs)
        { 
            int i = 0;
            int j = 0;
            int length1 = inputs[0].Length;
            int length2 = inputs.Length;
            double[] avg = new double[length1];
            double[] delta = new double[length1];
            for (i = 0; i < length1; i++)
            {
                double aver = 0;
                for (j = 0; j < length2; j++)
                {
                    aver += inputs[j][i];
                }
                aver /= length2;
                avg[i] = aver;
            }
            for (i = 0; i < length1; i++)
            {
                double d = 0;
                for (j = 0; j < length2; j++)
                {
                    d += Math.Pow(inputs[j][i] - avg[i], 2);
                }
                d /= length2;
                delta[i] = d;
            }
            for (i = 0; i < length1; i++)
            {
                for (j = 0; j < length2; j++)
                {
                    double currentN = inputs[j][i];
                    inputs[j][i] = (currentN - avg[i]) / delta[i];
                }
            }
        }

        private void MinMax(double[][] dataset)
        {
            int length1 = dataset[0].Length;
            int length2 = dataset.Length;
            double[] max = new double[length1];
            double[] min = new double[length1];
            for(int i = 0; i < length1; i++)
            {
                for (int j = 0; j < length2; j++)
                {
                    double Data = dataset[j][i];
                    if (Data > max[i])
                    {
                        max[i] = Data;
                    }
                    if (Data < min[i])
                    {
                        min[i] = Data;
                    }
                }
            }

            for (int i = 0; i < length1; i++)
            {
                for (int j = 0; j < length2; j++)
                {
                    dataset[j][i] = (dataset[j][i] - min[i]) / (max[i] - min[i]);
                }
            }
        }

        private void InitializeNeurons()
        {
            List<Neuron> currentLayer = new List<Neuron>(Topology.InputCount);
            int WeightsCount = 1;
            Random random = new Random();
            for (int i = 0; i < Topology.InputCount; i++)
            {
                currentLayer.Add(new Neuron(NeuronType.Input, WeightsCount, 0, i));
            }
            neuronLayer.Add(new Layer(currentLayer));
            for (int i = 0; i < Topology.HiddenCounts.Count; i++)
            {
                WeightsCount = neuronLayer.Last().Count;
                currentLayer = new List<Neuron>(Topology.HiddenCounts[i]);
                for (int j = 0; j < Topology.HiddenCounts[i]; j++)
                {
                    currentLayer.Add(new Neuron(NeuronType.Hidden, WeightsCount, i + 1, j, random));
                }
                neuronLayer.Add(new Layer(currentLayer));
            }
            currentLayer = new List<Neuron>(Topology.OutputCount);
            WeightsCount = neuronLayer.Last().Count;
            for (int i = 0; i < Topology.OutputCount; i++)
            {
                currentLayer.Add(new Neuron(NeuronType.Output, WeightsCount, Topology.Count - 1, i, random));
            }
            neuronLayer.Add(new Layer(currentLayer));
        }
        /*public double Backpropagation(double[] inputs, double expected)
        {
            EducationNeuron educate = new EducationNeuron();
            double error = FeedForward(inputs) - expected;
            List<double> outputs = null;
            int count = neuronLayer.Count;
            Layer currentLayer = neuronLayer[count - 1];
            Layer previousLayer = neuronLayer[count - 2];
            outputs = previousLayer.GetSignals();
            for (int j = 0; j < currentLayer.Count; j++)
            {
                Neuron currentNeuron = currentLayer[j];
                currentNeuron.Delta = educate.BackPropagation(currentNeuron, outputs, error, Topology.LearningRate);
            }

            for (int i = neuronLayer.Count - 2; i >= 1; i--)
            {
                currentLayer = neuronLayer[i];
                Layer forwardLayer = neuronLayer[i + 1];
                previousLayer = neuronLayer[i - 1];

                outputs = previousLayer.GetSignals();
                for (int j = 0; j < currentLayer.Count; j++)
                {
                    double deltaSum = 0;
                    Neuron currentNeuron = currentLayer[j];
                    double currentOutput = currentLayer[j].Output;
                    for (int k = 0; k < forwardLayer.Count; k++)
                    {
                        double delta = forwardLayer.Neurons[k].Delta;
                        double weights = forwardLayer.Neurons[k].Weights[j];
                        deltaSum += delta * weights;
                    }
                    deltaSum *= currentOutput;
                    currentNeuron.Delta = educate.BackPropagation(currentNeuron, outputs, deltaSum, Topology.LearningRate);
                }
            }
            return error;
        }
        public double Backpropagation(List<double> inputs, double expected)
        {
            EducationNeuron educate = new EducationNeuron();
            double error = FeedForward(inputs);
            List<double> outputs = null;
            for (int i = neuronLayer.Count - 1; i > neuronLayer.Count - 2; i--)
            {
                Layer currentLayer = neuronLayer[i];
                Layer previousLayer = neuronLayer[i - 1];
                outputs = previousLayer.GetSignals();
                for (int j = 0; j < currentLayer.Count; j++)
                {
                    Neuron currentNeuron = currentLayer[j];
                    currentNeuron.Delta = educate.BackPropagation(currentNeuron, outputs, error, Topology.LearningRate);
                }
            }
            double deltaSum = error;
            for (int i = neuronLayer.Count - 2; i >= 1; i--)
            {
                Layer currentLayer = neuronLayer[i];
                Layer forwardLayer = neuronLayer[i + 1];
                Layer previousLayer = neuronLayer[i - 1];

                outputs = previousLayer.GetSignals();
                for (int j = 0; j < currentLayer.Count; j++)
                {
                    for (int k = 0; k < forwardLayer.Count; k++)
                    {
                        double delta = forwardLayer.Neurons[k].Delta;
                        double output = forwardLayer.Neurons[k].Output;
                        double weights = forwardLayer.Neurons[k].Weights[j];
                        deltaSum += delta * output * weights;
                    }
                    Neuron currentNeuron = currentLayer[j];
                    currentNeuron.Delta = educate.BackPropagation(currentNeuron, outputs, error, Topology.LearningRate);
                }
            }
            return error;
        }
*/

    }
}
