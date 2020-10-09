import tensorflow as tf
from utils import double_flip_sample, single_flip_sample
from math import pi


class RNNWaveFunction:
    '''Creates RNN wave function.
    Args:
        rnn_cell: keras rnn cell
        ffnn: keras neural net that processes the hidden state of a RNN
        number_of_particles: int value, number of particles in a system
        local_dim: int value, dimension of local Hilbert spaces'''

    def __init__(self, rnn_cell, ffnn,
                 number_of_particles, local_dim):
     
        self.cell = rnn_cell  # rnn cell
        self.nn = ffnn  # neural network that processes output of rnn
        self.N = number_of_particles  # number of particles
        self.n = local_dim  # dimension of local Hilbert spaces
        self.state_size = rnn_cell.state_size  # size of hidden state

        # full NN that prcoesses samples
        inp = tf.keras.Input((number_of_particles, local_dim))
        in_state = tf.keras.Input((self.state_size,))
        rnn_out = tf.keras.layers.RNN(rnn_cell, return_sequences=True,)(inp, initial_state=in_state)
        nn_out = ffnn(rnn_out)
        nn_out_resh = tf.keras.layers.Reshape((self.N, self.n, 2))(nn_out)
        logits = nn_out_resh[..., 0]
        phase = nn_out_resh[..., 1]
        log_p = tf.keras.layers.Lambda(lambda x: tf.nn.log_softmax(x, axis=-1))(logits)
        phi = pi * tf.keras.activations.softsign(phase)
        self.rnn = tf.keras.Model(inputs=[inp, in_state], outputs=[log_p, phi])

    @tf.function
    def sample(self, number_of_samples):
        '''Create set of samples from |psi|^2.
        Args:
            number_of_samples: int value, number of samples to create
        Returns:
            float tensor of shape
            (number_of_samples, number_of_particles, dim_of_local_space),
            one hot representation of samples from |psi|^2'''

        # initial state
        in_state = tf.ones((number_of_samples, self.state_size))
        # initial samples tensor
        samples = tf.ones((number_of_samples, 1, self.n))
        # initial number of iteration
        i = tf.constant(0)

        # body of a loop
        def sample_one_side(i, state, samples):
            # Apply rnn cell to previous sample and state, output shape
            # (number_of_samples, self.state_size)
            cell_out, _ = self.cell(inputs=samples[:, -1], states=state)
            # apply NN to output of rnn cell, output shape
            # (number_of_samples, 2 * self.n)
            nn_out = self.nn(cell_out)
            # reshape of NN output, output shape
            # (number_of_samples, self.n, 2)
            nn_out_resh = tf.reshape(nn_out, (number_of_samples, self.n, 2))
            # generate sample using Gumbel trick,
            # output shape (number_of_samples, dim_of_local_space)
            logits = nn_out_resh[..., 0]
            log_p = tf.nn.log_softmax(logits, axis=-1)
            eps = tf.random.uniform(log_p.shape)
            eps = -tf.math.log(-tf.math.log(eps))
            s = tf.one_hot(tf.argmax(log_p + eps, axis=-1), axis=-1, depth=self.n)
            # add generated sample to samples
            s = s[:, tf.newaxis]
            new_samples = tf.concat([samples, s], axis=1)
            return i+1, cell_out, new_samples
        # condition
        cond = lambda i, state, samples: i < self.N
        # samples generation
        _, _, samples = tf.while_loop(cond,
                                      sample_one_side,
                                      loop_vars=[i, in_state, samples],
                                      shape_invariants=[i.get_shape(),
                                                        in_state.get_shape(),
                                                        tf.TensorShape((number_of_samples, None, self.n))])
        return samples[:, 1:]

    @tf.function
    def value(self, samples):
        '''Calculates value of psi function in a given points.
        Args:
            samples: float tensor of shape
                (number_of_samples, number_of_particles, dim_of_local_space),
                point, where one needs to calculate values of psi function
        Returns:
            list of two tensors of shape (number_of_samples,),
            first tensor is log(|psi|^2) second is phase of a
            psi function'''

        # initial state
        state = tf.ones((samples.shape[0], self.state_size))
        # initial sample
        s0 = tf.ones((samples.shape[0], 1, self.n))
        log_p, phi = self.rnn([tf.concat([s0, samples[:, :-1]], axis=1), state])
        log_p = tf.reduce_sum(log_p * samples, axis=(-2, -1))
        phi = tf.reduce_sum(phi * samples, axis=(-2, -1))
        return log_p, phi

    @tf.function
    def local_energy(self,
                     connections,
                     ampls,
                     local_fields,
                     samples):
        '''Returns value of a local energy in geven points.
        Args:
            connections: int tensor of shape (number_of_connections, 2),
                first index enumerates number of connections, second
                index enumerates sites that are connected
            ampls: complex valued tensor of shape (number_of_connections, 3),
                amplitudes of xx, yy and zz terms between connected sites
            local_fields: complex valued tensor of shape
                (number_of_particles, 3), x, y, z components of external
                magnetic field per site
            samples: float tensor of shape 
                (number_of_samples, number_of_particles, dim_of_local_space),
                point, where one needs to calculate values of local energy
        Returns:
            complex value tensor of shape (number_of_samples,), local
            energy per sample'''

        # denominator
        log_p, phi = self.value(samples)
        log_p = tf.cast(log_p, dtype=tf.complex64)
        phi = tf.cast(phi, dtype=tf.complex64)
        
        # complex version of samples
        csamples = tf.cast(samples, dtype=tf.complex64)

        # pauli matrices assotiated multipliers
        x = tf.constant([1, 1], dtype=tf.complex64)
        y = tf.constant([-1j, 1j], dtype=tf.complex64)
        z = tf.constant([1, -1], dtype=tf.complex64)
        
        # initial value of local energy
        E = tf.zeros((samples.shape[0],), dtype=tf.complex64)
        
        # first tf while loop that calculates contrbution of interaction into
        # local energy
        
        # initial value of counter
        iter = tf.constant(0)
        
        # contribution of one interaction term
        def local_int_term(iter, E):

            ind = connections[iter]
            # xx energy
            Exx = tf.reduce_sum(csamples[:, ind[0]] * x, axis=-1) *\
            tf.reduce_sum(csamples[:, ind[1]] * x, axis=-1)
            Exx = Exx * ampls[iter, 0]
            # yy energy
            Eyy = tf.reduce_sum(csamples[:, ind[0]] * y, axis=-1) *\
            tf.reduce_sum(csamples[:, ind[1]] * y, axis=-1)
            Eyy = Eyy * ampls[iter, 1]
            # zz energy
            Ezz = tf.reduce_sum(csamples[:, ind[0]] * z, axis=-1) *\
            tf.reduce_sum(csamples[:, ind[1]] * z, axis=-1)
            Ezz = Ezz * ampls[iter, 2]

            double_flipped_samples = double_flip_sample(samples, ind)
            double_flipped_log_p, double_flipped_phi = self.value(double_flipped_samples)
            double_flipped_log_p = tf.cast(double_flipped_log_p, dtype=tf.complex64)
            double_flipped_phi = tf.cast(double_flipped_phi, dtype=tf.complex64)
            ratio = tf.exp(0.5 * double_flipped_log_p -\
                           0.5 * log_p +\
                           1j * double_flipped_phi -\
                           1j * phi)
            E_new = Exx * ratio + Eyy * ratio + Ezz
            return iter + 1, E_new + E

        # stopping criteria
        cond1 = lambda iter, E: iter < connections.shape[0]
        # loop
        _, E = tf.while_loop(cond1, local_int_term, loop_vars=[iter, E])

        # second tf while loop that calculates contrbution of ext. magnetic
        # fields into the local energy
        def local_term(iter, E):
            # x energy
            Ex = tf.reduce_sum(csamples[:, iter] * x, axis=-1)
            Ex = Ex * local_fields[iter, 0]
            # y energy
            Ey = tf.reduce_sum(csamples[:, iter] * x, axis=-1)
            Ey = Ey * local_fields[iter, 1]
            # z energy
            Ez = tf.reduce_sum(csamples[:, iter] * x, axis=-1)
            Ez = Ez * local_fields[iter, 2]

            single_flipped_sample = single_flip_sample(samples, iter)
            single_flipped_log_p, single_flipped_phi = self.value(single_flipped_sample)
            single_flipped_log_p = tf.cast(single_flipped_log_p, dtype=tf.complex64)
            single_flipped_phi = tf.cast(single_flipped_phi, dtype=tf.complex64)
            ratio = tf.exp(0.5 * single_flipped_log_p -\
                           0.5 * log_p +\
                           1j * single_flipped_phi -\
                           1j * phi)
            E_new = Ex * ratio + Ey * ratio + Ez
            return iter + 1, E_new + E

        # stopping criteria
        cond2 = lambda iter, E: iter < self.N

        # loop
        _, E = tf.while_loop(cond2, local_term, loop_vars=[iter, E])
        return E

    def correlator(self, ind1, ind2, n):
        """Returnes correlator build from samples.
        Args:
            ind1: int value, position of the first site
            ind2: int value, position of the second site
            n: int value, n*10000 samples are used for averaging
        Return:
            complex valued tensor of shape (3,), xx, yy, zz
            correlators values"""

        # pauli matrices assotiated multipliers
        x = tf.constant([1, 1], dtype=tf.complex64)
        y = tf.constant([-1j, 1j], dtype=tf.complex64)
        z = tf.constant([1, -1], dtype=tf.complex64)
        corr = tf.constant([0, 0, 0], dtype=tf.complex64)
        for _ in range(n):
            # denominator
            samples = self.sample(10000)
            log_p, phi = self.value(samples)
            log_p = tf.cast(log_p, dtype=tf.complex64)
            phi = tf.cast(phi, dtype=tf.complex64)
            
            # complex version of samples
            csamples = tf.cast(samples, dtype=tf.complex64)
            
            # xx corr
            corrxx = tf.reduce_sum(csamples[:, ind1] * x, axis=-1) *\
            tf.reduce_sum(csamples[:, ind2] * x, axis=-1)
            # yy corr
            corryy = tf.reduce_sum(csamples[:, ind1] * y, axis=-1) *\
            tf.reduce_sum(csamples[:, ind2] * y, axis=-1)
            # zz energy
            corrzz = tf.reduce_sum(csamples[:, ind1] * z, axis=-1) *\
            tf.reduce_sum(csamples[:, ind2] * z, axis=-1)

            double_flipped_samples = double_flip_sample(samples, tf.constant([ind1, ind2]))
            double_flipped_log_p, double_flipped_phi = self.value(double_flipped_samples)
            double_flipped_log_p = tf.cast(double_flipped_log_p, dtype=tf.complex64)
            double_flipped_phi = tf.cast(double_flipped_phi, dtype=tf.complex64)
            ratio = tf.exp(0.5 * double_flipped_log_p -\
                           0.5 * log_p +\
                           1j * double_flipped_phi -\
                           1j * phi)
            
            corr_update = tf.concat([tf.reduce_mean(corrxx * ratio),
                                     tf.reduce_mean(corryy * ratio),
                                     tf.reduce_mean(corrzz)], axis=0)
            corr = corr + corr_update
        return corr / n

    @tf.function
    def train(self,
              sample_size,
              number_of_iters,
              opt,
              connections,
              ampls,
              local_fields):
        """Optimizes RNN Wavefunction.
        Args:
            sample_size: int value, number of samples, that is used to
                calculate expectation values
            number_of_iters: in value, number of iterations
            opt: tf optimizer
            connections: int tensor of shape (number_of_connections, 2),
                first index enumerates number of connections, second
                index enumerates sites that are connected
            ampls: complex valued tensor of shape (number_of_connections, 3),
                amplitudes of xx, yy and zz terms between connected sites
            local_fields: complex valued tensor of shape
                (number_of_particles, 3), x, y, z components of external
                magnetic field per site"""

        av_E = tf.constant(0, dtype=tf.complex64)
        E = tf.constant([0], dtype=tf.float32)
        iter = tf.constant(0)
        # body of a loop
        def train_step(E, av_E, iter):
            with tf.GradientTape() as tape:
                samples = tf.stop_gradient(self.sample(sample_size))
                local_E = tf.stop_gradient(self.local_energy(connections,
                                                            ampls,
                                                            local_fields,
                                                            samples))
                log_p, phi = self.value(samples)
                log_p = tf.cast(log_p, dtype=tf.complex64)
                phi = tf.cast(phi, dtype=tf.complex64)
                log_psi_conj = 0.5 * log_p - 1j * phi
                loss = 2 * tf.math.real(tf.reduce_mean(log_psi_conj * (local_E - av_E)))
            av_E = tf.reduce_mean(local_E)
            E = tf.concat([E, av_E], axis=0)
            grad = tape.gradient(loss, self.ffnn.weights + self.cell.weights)
            opt.apply_gradients(zip(grad, self.ffnn.weights + self.cell.weights))
            return E, av_E, iter+1
        # stopping criteria
        cond = lambda E, av_E, iter: iter < number_of_iters
        # loop
        E, _, _ = tf.while_loop(cond, train_step,
                                      loop_vars=[E, av_E, iter],
                                      shape_invariants=[tf.TensorShape((None,)),
                                                        av_E.shape,
                                                        iter.shape])
        return E[1:]
