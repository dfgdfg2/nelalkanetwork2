using ConsoleApp1;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    public abstract class IOptimizer
    {
        public double Epoch = 1;
        public AbRegularizationWeights regul;
        public IOptimizer(AbRegularizationWeights regul = null)
        {
            if (regul == null)
            {
                this.regul = new L0();
                return;
            }
            this.regul = regul;
        }
        public abstract void BackPropagation(Neuron neuron, List<double> outputs, double learningRate, int epoch/*, double error*/);
        public virtual List<double> GetDeltaWeights(double learningRate, double delta, List<Neuron> outputs, int ind = 0)
        {
            List<double> weights = new List<double>();
            for (int i = 0; i < outputs.Count; i++)
            {
                weights.Add(learningRate * delta * outputs[i].Output);
            }
            return weights;
        }
        public virtual int SetWeights(Neuron neuron, List<double> weights)
        {
            int i = 0;
            for (; i < neuron.WeightsCount; i++)
            {
                double weight = neuron.Weights[i];
                neuron.Weights[i] = weight - weights[i] - regul.Regulate(weight);
            }
            return i;
        }

        public double GetFromAllDelta(Layer forwardLayer, int j)
        {
            double deltaSum = 0;
            for (int k = 0; k < forwardLayer.Count; k++)
            {
                Neuron neuron = forwardLayer.Neurons[k];
                double delta = neuron.Delta;
                double weights = neuron.Weights[j];
                deltaSum += delta * weights;
            }
            return deltaSum;
        }
    }
    /// <summary>
    /// ////////////////////////////
    /// </summary>
    public class DefaultOptimizer : IOptimizer
    {
        public DefaultOptimizer(AbRegularizationWeights regul = null) : base(regul)
        {

        }
        public override void BackPropagation(Neuron neuron, List<double> outputs, double learningRate, int epoch/*, double delta*/)
        {
            //neuron.Delta = delta;
            double delta = neuron.Delta;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                double output = outputs[i];
                double currentWeight = neuron.Weights[i];
                double regulator = regul.Regulate(currentWeight); 
                neuron.Weights[i] = currentWeight - (learningRate * delta * output - regulator);
            }
        }
    }
    /// <summary>
    /// ////////////////////////////
    /// </summary>
    public class Momentum : IOptimizer
    {
        protected double[][][] Inertions { get; set; }
        protected double Koef { get; set; }
        public Momentum(Topology topology, AbRegularizationWeights regul = null) : base(regul)
        {
            Inertions = new double[topology.Count - 1][][];
            for (int i = 0; i < topology.CollectionCounts.Count - 1; i++)
            {
                int collection = topology.CollectionCounts[i + 1];
                Inertions[i] = new double[collection][];
                int weights = topology.CollectionCounts[i];
                for (int j = 0; j < collection; j++)
                {
                    Inertions[i][j] = new double[weights];
                }
            }
            Koef = 0.9;
        }

        public override void BackPropagation(Neuron neuron, List<double> outputs, double learningRate, int epoch/*, double delta*/)
        {
            //neuron.Delta = delta;
            double delta = neuron.Delta;
            int numberLayer = neuron.NumberLayer;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                double output = outputs[i];
                double currentWeight = neuron.Weights[i];
                double regulator = regul.Regulate(currentWeight);
                double inertia = Inertions[numberLayer - 1][neuron.NumberOfLayer][i] * Koef + delta * learningRate * output; // исправить error на delta в случае нестабильности!!!!!!!!!!!
                Inertions[numberLayer - 1][neuron.NumberOfLayer][i] = inertia;
                neuron.Weights[i] = currentWeight - inertia - regulator;
            }
        }
        public override List<double> GetDeltaWeights(double learningRate, double delta, List<Neuron> outputs, int ind = 0)
        {
            List<double> inertia = new List<double>();
            int numberLayer = outputs[0].NumberLayer;
            for (int i = 0; i < outputs.Count; i++)
            {
                inertia.Add(0);
                inertia[i] = Inertions[numberLayer][ind][i] * Koef + delta * learningRate * outputs[i].Output;
                Inertions[numberLayer][ind][i] = inertia[i];
            }
            return inertia;
        }
    }
    //NOT DEFINED
    public class NesterovMomentum : Momentum
    {
        public NesterovMomentum(Topology topology, AbRegularizationWeights regul = null) : base(topology, regul)
        {

        }

        public override void BackPropagation(Neuron neuron, List<double> outputs, double learningRate, int epoch/*, double delta*/)
        {
            //neuron.Delta = delta;
            double delta = neuron.Delta;
            int numberLayer = neuron.NumberLayer;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                double output = outputs[i];
                double currentWeight = neuron.Weights[i];
                double regulator = regul.Regulate(currentWeight);
                double inertia = Inertions[numberLayer - 1][neuron.NumberOfLayer][i];
                double predictedWeight = currentWeight - inertia * Koef;
                //inertia = inertia * Koef + delta * learningRate * previousOutputs[i];
                inertia = inertia * Koef + delta * learningRate * (output - Koef * inertia);
                Inertions[numberLayer - 1][neuron.NumberOfLayer][i] = inertia;
                neuron.Weights[i] = predictedWeight - inertia - regulator;
            }
        }

        public override List<double> GetDeltaWeights(double learningRate, double delta, List<Neuron> outputs, int ind = 0)
        {
            List<double> inertia = new List<double>();
            int numberLayer = outputs[0].NumberLayer;
            for (int i = 0; i < outputs.Count; i++)
            {
                inertia.Add(0);
                double cur = Inertions[numberLayer][ind][i];
                inertia[i] = cur * Koef + delta * learningRate * (outputs[i].Output - Koef * cur);
                Inertions[numberLayer][ind][i] = inertia[i];
            }
            return inertia;
        }

        public override int SetWeights(Neuron neuron, List<double> weights)
        {
            int i = 0;
            for (; i < neuron.WeightsCount; i++)
            {
                double predictedWeight = neuron.Weights[i] - Inertions[neuron.NumberLayer - 1][neuron.NumberOfLayer][i] * Koef;
                neuron.Weights[i] = predictedWeight - weights[i] - regul.Regulate(neuron.Weights[i]);
            }
            return i;
        }
    }

    public class RMSProp : IOptimizer
    {
        protected double[][][] G { get; set; }
        double E = Math.Pow(10, -8);
        double Alpha = 0.85;
        public RMSProp(Topology topology, AbRegularizationWeights regul = null) : base(regul)
        {
            G = new double[topology.Count - 1][][];
            for (int i = 0; i < topology.CollectionCounts.Count - 1; i++)
            {
                int collection = topology.CollectionCounts[i + 1];
                G[i] = new double[collection][];
                int weights = topology.CollectionCounts[i];
                for (int j = 0; j < collection; j++)
                {
                    G[i][j] = new double[weights];
                }
            }
        }

        public override void BackPropagation(Neuron neuron, List<double> outputs, double learningRate, int epoch/*, double delta*/)
        {
            //neuron.Delta = delta;
            double delta = neuron.Delta;
            int numberLayer = neuron.NumberLayer;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                double output = outputs[i];
                double currentWeight = neuron.Weights[i];
                double regulator = regul.Regulate(currentWeight);
                double g = G[numberLayer - 1][neuron.NumberOfLayer][i];
                g = Alpha * g + (1 - Alpha) * Math.Pow(delta * output, 2);
                G[numberLayer - 1][neuron.NumberOfLayer][i] = g;
                neuron.Weights[i] = currentWeight - (delta * output * learningRate / Math.Sqrt(g + E)) - regulator;
            }
        }

        public override List<double> GetDeltaWeights(double learningRate, double delta, List<Neuron> outputs, int ind = 0)
        {
            List<double> rms = new List<double>();
            int numberLayer = outputs[0].NumberLayer;
            for (int i = 0; i < outputs.Count; i++)
            {
                rms.Add(0);
                double output = outputs[i].Output;
                double g = G[numberLayer][ind][i];
                g = Alpha * g + (1 - Alpha) * Math.Pow(delta * output, 2);
                G[numberLayer][ind][i] = g;
                rms[i] = delta * output * learningRate / (Math.Sqrt(g) + E);
            }
            return rms;
        }
    }

    public class Adagrad : IOptimizer
    {
        protected double[][][] G { get; set; }
        double E = Math.Pow(10, -9);
        public Adagrad(Topology topology, AbRegularizationWeights regul = null) : base(regul)
        {
            G = new double[topology.Count - 1][][];
            for (int i = 0; i < topology.CollectionCounts.Count - 1; i++)
            {
                int collection = topology.CollectionCounts[i + 1];
                G[i] = new double[collection][];
                int weights = topology.CollectionCounts[i];
                for (int j = 0; j < collection; j++)
                {
                    G[i][j] = new double[weights];
                }
            }
        }

        public override void BackPropagation(Neuron neuron, List<double> outputs, double learningRate, int epoch/*, double delta*/)
        {
            //neuron.Delta = delta;
            double delta = neuron.Delta;
            int numberLayer = neuron.NumberLayer;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                double output = outputs[i];
                double currentWeight = neuron.Weights[i];
                double regulator = regul.Regulate(currentWeight);
                double g = G[numberLayer - 1][neuron.NumberOfLayer][i];
                double weightDelta = delta * output;
                double reg = Math.Pow(weightDelta, 2); 
                g = g + reg;
                neuron.Weights[i] = currentWeight - (weightDelta * learningRate / Math.Sqrt(g + E)) - regulator;
                G[numberLayer - 1][neuron.NumberOfLayer][i] = g;
            }
        }

        public override List<double> GetDeltaWeights(double learningRate, double delta, List<Neuron> outputs, int ind = 0)
        {
            List<double> ada = new List<double>();
            int numberLayer = outputs[0].NumberLayer;
            for (int i = 0; i < outputs.Count; i++)
            {
                ada.Add(0);
                double output = outputs[i].Output;
                double g = G[numberLayer][ind][i];
                double weightDelta = delta * output;
                double reg = Math.Pow(weightDelta, 2);
                g = g + reg;
                G[numberLayer][ind][i] = g;
                ada[i] = weightDelta * learningRate / Math.Sqrt(g + E);
                G[numberLayer][ind][i] = g;
            }
            return ada;
        }
    }
    public class Adam : IOptimizer
    {
        double[][][] M { get; set; }
        double[][][] V { get; set; }
        double E = Math.Pow(10, -8);
        double Beta1 = 0.9;
        double Beta2 = 0.999;
        public Adam(Topology topology, AbRegularizationWeights regul = null) : base(regul)
        {
            M = new double[topology.Count - 1][][];
            V = new double[topology.Count - 1][][];
            for (int i = 0; i < topology.CollectionCounts.Count - 1; i++)
            {
                int collection = topology.CollectionCounts[i + 1];
                M[i] = new double[collection][];
                V[i] = new double[collection][];
                int weights = topology.CollectionCounts[i];
                for (int j = 0; j < collection; j++)
                {
                    M[i][j] = new double[weights];
                    V[i][j] = new double[weights];
                }
            }
        }

        public override void BackPropagation(Neuron neuron, List<double> outputs, double learningRate, int epoch/*, double delta*/)
        {
            //neuron.Delta = delta;
            double delta = neuron.Delta;
            int numberLayer = neuron.NumberLayer;
            for (int i = 0; i < neuron.WeightsCount; i++)
            {
                double output = outputs[i];
                double currentWeight = neuron.Weights[i];
                double regulator = regul.Regulate(currentWeight);
                double v = V[numberLayer - 1][neuron.NumberOfLayer][i];
                double m = M[numberLayer - 1][neuron.NumberOfLayer][i];
                m = (Beta1 * m) + (1 - Beta1) * (delta * output);
                v = (Beta2 * v) + (1 - Beta2) * Math.Pow(delta * output, 2);
                double m_corr = m / (1 - Math.Pow(Beta1, epoch));
                double v_corr = v / (1 - Math.Pow(Beta2, epoch));
                neuron.Weights[i] = currentWeight - (learningRate * m_corr) / Math.Sqrt(v_corr + E) - regulator;
                V[numberLayer - 1][neuron.NumberOfLayer][i] = v;
                M[numberLayer - 1][neuron.NumberOfLayer][i] = m;
            }
        }

        public override List<double> GetDeltaWeights(double learningRate, double delta, List<Neuron> outputs, int ind = 0)
        {
            List<double> adam = new List<double>();
            int numberLayer = outputs[0].NumberLayer;
            for (int i = 0; i < outputs.Count; i++)
            {
                double output = outputs[i].Output;
                adam.Add(0);
                double v = V[numberLayer][ind][i];
                double m = M[numberLayer][ind][i];
                m = (Beta1 * m) + (1 - Beta1) * (delta * output);
                v = (Beta2 * v) + (1 - Beta2) * Math.Pow(delta * output, 2);
                double m_corr = m / (1 - Math.Pow(Beta1, Epoch));
                double v_corr = v / (1 - Math.Pow(Beta2, Epoch));
                adam[i] = (learningRate * m_corr) / Math.Sqrt(v_corr + E);
                V[numberLayer][ind][i] = v;
                M[numberLayer][ind][i] = m;
            }
            return adam;
        }
    }
}
