using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    public abstract class AbGradientDescent
    {
        protected NeuralNetwork currentNN;
        public ILearnRegularization regularization { get; private set; }
        protected List<Layer> neuronLayer;
        protected Topology topology;
        protected IOptimizer Optimizer;
        protected List<Regularizations> regularizations;
        protected int CurrentEpoch;
        protected DelegateforWeights regulatorW;
        public abstract double Backpropagation<T>(T dataset, int[] expected, int epoch);
        public AbGradientDescent(NeuralNetwork nn, IOptimizer optimizer = null)
        {
            currentNN = nn;
            neuronLayer = currentNN.neuronLayer;
            topology = currentNN.Topology;
            if (optimizer == null)
            {
                optimizer = new DefaultOptimizer();
                Optimizer = optimizer;
            }
            else
            {
                Optimizer = optimizer;
            }
            currentNN.Reg = new DefaultRegularization();
            regularization = new DefaultRegularization();
            regulatorW = (double d) => { return 0; };
            if (Optimizer.regulList != null && Optimizer.regulList.Count != 0)
            {
                Regularization regu = new Regularization();
                int IsComb = 0;
                for (int i = 0; i < Optimizer.regulList.Count; i++)
                {
                    Regularizations reg = Optimizer.regulList[i];
                    switch (reg)
                    {
                        case Regularizations.L1:
                            regulatorW = regu.L1;
                            IsComb++;
                            break;
                        case Regularizations.L2:
                            regulatorW = regu.L2;
                            IsComb++;
                            break;
                        case Regularizations.Dropout:
                            regularization = new RegDropoutLearn();
                            currentNN.Reg = new RegDropoutTest();
                            break;
                        /*case Regularizations.BatchNormalization:
                            regularization = new BatchNormalizationLearn();
                            currentNN.Reg
                            break;
                        */
                    }
                }
                if (IsComb == 2)
                {
                    regulatorW = regu.L1andL2;
                }
            }
        }
    }

    public class MiniBatchGD : AbGradientDescent
    {
        Random random;
        double LearningRate;
        int Capacity;
        public MiniBatchGD(NeuralNetwork nn, int capacity = 2, IOptimizer optimizer = null) : base(nn, optimizer) 
        {
            Capacity = capacity; 
            random = new Random();
        }

        public override double Backpropagation<T>(T dataset, int[] expected, int epoch)
        {
            if (dataset is List<double> outputsL)
            {
                double error = 0;
                for (int i = 0; i < epoch; i++)
                {
                    CurrentEpoch++;
                    error += Backpropagation(outputsL, expected[0]);
                }
                return error;
            }
            else if (dataset is double[][] outputsM)
            {
                double error = 0;
                for (int i = 0; i < epoch; i++)
                {
                    CurrentEpoch++;
                    for (int j = 0; j < Capacity; j++)
                    {
                        int rnd = random.Next(0, expected.Length - 1);
                        int ex = expected[rnd];
                        error += Backpropagation(outputsM[rnd], ex);
                    }
                }
                return error * Capacity / epoch;
            }
            else
            {
                return 0;
            }
        }

        public double Backpropagation(List<double> dataset, int expected)
        {
            double error = regularization.FeedForward(dataset, currentNN.neuronLayer) - expected;
            int count = neuronLayer.Count;
            Layer currentLayer = neuronLayer[count - 1];
            Layer previousLayer = neuronLayer[count - 2];
            List<double> outputs = previousLayer.GetSignals();
            LearningRate = topology.LearningRate / Math.Pow(CurrentEpoch, 0.3);

            for (int j = 0; j < currentLayer.Count; j++)
            {
                Neuron currentNeuron = currentLayer[j];
                currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, error, LearningRate, CurrentEpoch, regulatorW);
            }

            for (int i = neuronLayer.Count - 2; i >= 1; i--)
            {
                currentLayer = neuronLayer[i];
                Layer forwardLayer = neuronLayer[i + 1];
                previousLayer = neuronLayer[i - 1];

                outputs = previousLayer.GetSignals();
                for (int j = 0; j < currentLayer.Count; j++)
                {
                    Neuron currentNeuron = currentLayer[j];
                    double deltaSum = Optimizer.GetFromAllDelta(forwardLayer, j);
                    currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, deltaSum, LearningRate, CurrentEpoch, regulatorW);
                }
            }
            return error * error;
        }

        public double Backpropagation(double[] dataset, int expected)
        {
            double error = regularization.FeedForward(dataset, currentNN.neuronLayer) - expected;
            int count = neuronLayer.Count;
            Layer currentLayer = neuronLayer[count - 1];
            Layer previousLayer = neuronLayer[count - 2];
            List<double> outputs = previousLayer.GetSignals();
            for (int j = 0; j < currentLayer.Count; j++)
            {
                Neuron currentNeuron = currentLayer[j];
                currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, error, topology.LearningRate, CurrentEpoch, regulatorW);
            }

            for (int i = neuronLayer.Count - 2; i >= 1; i--)
            {
                currentLayer = neuronLayer[i];
                Layer forwardLayer = neuronLayer[i + 1];
                previousLayer = neuronLayer[i - 1];

                outputs = previousLayer.GetSignals();
                for (int j = 0; j < currentLayer.Count; j++)
                {
                    Neuron currentNeuron = currentLayer[j];
                    double deltaSum = Optimizer.GetFromAllDelta(forwardLayer, j);
                    currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, deltaSum, topology.LearningRate, CurrentEpoch, regulatorW);
                }
            }
            return error * error;
        }
    }

    public class StochasticGradient : AbGradientDescent
    {
        Random random;
        double LearningRate;
        public StochasticGradient(NeuralNetwork nn, IOptimizer optimizer = null) : base(nn, optimizer)
        {
            random = new Random();
        }

        public override double Backpropagation<T>(T dataset, int[] expected, int epoch)
        {
            if (dataset is List<double> outputsL)
            {
                double error = 0;
                for (int i = 0; i < epoch; i++)
                {
                    CurrentEpoch++;
 
                    error += Backpropagation(outputsL, expected[0]);
                }
                return error * error;
            }
            else if (dataset is double[][] outputsM)
            {
                double error = 0;
                for (int i = 0; i < epoch; i++)
                {
                    CurrentEpoch++;
                    int rnd = random.Next(0, expected.Length - 1);
                    int ex = expected[rnd];
                    error += Backpropagation(outputsM[rnd], ex);
                }
                return error * outputsM.Length / epoch;
            }
            else
            {
                return 0;
            }
        }

        public double Backpropagation(List<double> dataset, int expected)
        {
            double error = regularization.FeedForward(dataset, currentNN.neuronLayer) - expected;
            int count = neuronLayer.Count;
            Layer currentLayer = neuronLayer[count - 1];
            Layer previousLayer = neuronLayer[count - 2];
            List<double> outputs = previousLayer.GetSignals();
            outputs = previousLayer.GetSignals();

            LearningRate = topology.LearningRate / Math.Pow(CurrentEpoch, 0.1);

            for (int j = 0; j < currentLayer.Count; j++)
            {
                Neuron currentNeuron = currentLayer[j];
                currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, error, LearningRate, CurrentEpoch, regulatorW);
            }

            for (int i = neuronLayer.Count - 2; i >= 1; i--)
            {
                currentLayer = neuronLayer[i];
                Layer forwardLayer = neuronLayer[i + 1];
                previousLayer = neuronLayer[i - 1];

                outputs = previousLayer.GetSignals();
                for (int j = 0; j < currentLayer.Count; j++)
                {
                    Neuron currentNeuron = currentLayer[j];
                    double deltaSum = Optimizer.GetFromAllDelta(forwardLayer, j);
                    currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, deltaSum, LearningRate, CurrentEpoch, regulatorW);
                }
            }
            return error * error;
        }

        public double Backpropagation(double[] dataset, int expected)
        {
            double error = regularization.FeedForward(dataset, currentNN.neuronLayer) - expected;
            int count = neuronLayer.Count;
            Layer currentLayer = neuronLayer[count - 1];
            Layer previousLayer = neuronLayer[count - 2];
            List<double> outputs = previousLayer.GetSignals();
            outputs = previousLayer.GetSignals();

            LearningRate = topology.LearningRate / Math.Pow(CurrentEpoch, 0.1);

            for (int j = 0; j < currentLayer.Count; j++)
            {
                Neuron currentNeuron = currentLayer[j];
                currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, error, LearningRate, CurrentEpoch, regulatorW);
            }

            for (int i = neuronLayer.Count - 2; i >= 1; i--)
            {
                currentLayer = neuronLayer[i];
                Layer forwardLayer = neuronLayer[i + 1];
                previousLayer = neuronLayer[i - 1];

                outputs = previousLayer.GetSignals();
                for (int j = 0; j < currentLayer.Count; j++)
                {
                    Neuron currentNeuron = currentLayer[j];
                    double deltaSum = Optimizer.GetFromAllDelta(forwardLayer, j);
                    currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, deltaSum, LearningRate, CurrentEpoch, regulatorW);
                }
            }
            return error * error;
        }
    }

    public class GradientDescent : AbGradientDescent
    {
        public GradientDescent(NeuralNetwork nn, IOptimizer optimizer = null) : base(nn, optimizer)
        {
        }

        public override double Backpropagation<T>(T dataset, int[] expected, int epoch)
        {
            if (dataset is List<double> outputsL)
            {
                double error = 0;
                for (int i = 0; i < epoch; i++)
                {
                    CurrentEpoch++;
                    error += Backpropagation(outputsL, expected[0]);
                }
                return error;
            }
            else if (dataset is double[][] outputsM)
            {
                double error = 0;
                for (int j = 0; j < epoch; j++)
                {
                    CurrentEpoch++;
                    for (int i = 0; i < outputsM.Length; i++)
                    {
                        error += Backpropagation(outputsM[i], expected[i]);
                    }
                }
                return error / epoch;
            }
            else
            {
                return 0;
            }
        }

        public double Backpropagation(List<double> dataset, int expected)
        {
            double error = regularization.FeedForward(dataset, currentNN.neuronLayer) - expected;
            int count = neuronLayer.Count;
            Layer currentLayer = neuronLayer[count - 1];
            Layer previousLayer = neuronLayer[count - 2];
            List<double> outputs = previousLayer.GetSignals();
            for (int j = 0; j < currentLayer.Count; j++)
            {
                Neuron currentNeuron = currentLayer[j];
                currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, error, topology.LearningRate, CurrentEpoch, regulatorW);
            }

            for (int i = neuronLayer.Count - 2; i >= 1; i--)
            {
                currentLayer = neuronLayer[i];
                Layer forwardLayer = neuronLayer[i + 1];
                previousLayer = neuronLayer[i - 1];

                outputs = previousLayer.GetSignals();
                for (int j = 0; j < currentLayer.Count; j++)
                {
                    Neuron currentNeuron = currentLayer[j];
                    double deltaSum = Optimizer.GetFromAllDelta(forwardLayer, j);
                    currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, deltaSum, topology.LearningRate, CurrentEpoch, regulatorW);
                }
            }
            return error * error;
        }

        public double Backpropagation(double[] dataset, int expected)
        {
            double error = regularization.FeedForward(dataset, currentNN.neuronLayer) - expected;
            int count = neuronLayer.Count;
            Layer currentLayer = neuronLayer[count - 1];
            Layer previousLayer = neuronLayer[count - 2];
            List<double> outputs = previousLayer.GetSignals();
            for (int j = 0; j < currentLayer.Count; j++)
            {
                Neuron currentNeuron = currentLayer[j];
                currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, error, topology.LearningRate, CurrentEpoch, regulatorW);
            }

            for (int i = neuronLayer.Count - 2; i >= 1; i--)
            {
                currentLayer = neuronLayer[i];
                Layer forwardLayer = neuronLayer[i + 1];
                previousLayer = neuronLayer[i - 1];

                outputs = previousLayer.GetSignals();
                for (int j = 0; j < currentLayer.Count; j++)
                {
                    Neuron currentNeuron = currentLayer[j];
                    double deltaSum = Optimizer.GetFromAllDelta(forwardLayer, j);
                    currentNeuron.Delta = Optimizer.BackPropagation(currentNeuron, outputs, deltaSum, topology.LearningRate, CurrentEpoch, regulatorW);
                }
            }
            return error * error;
        }
    }
}
