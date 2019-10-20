This is a variation on the [Weasel program](https://en.wikipedia.org/wiki/Weasel_program).

You have a population of 10000 individuals.

On each generation:

* The individuals are each measured for their closeness to the phrase
  *"Methinks It Is Like A Weasel"*. The closeness algorithm is to calculate
  the difference between each character in the individual and the
  corresponding character in the target phrase. These differences are used as
  a vector to calculate the scalar distance to the target. This is then cubed
  to make the target much more favourable.

* A new generation is created by *"mating"* individuals with each other to
  produce a new individual.

  "Mating" is:

  * **Choose** two individuals at random, but with a likelihood of being
    chosen in proportion to their closeness to the ideal phrase.
  * **Crossover**: for a random value 0 < n < individual.len, take the first n
    chars from the first individual and the remaining chars from the second
    individual.
  * **Mutation**: if some random threshold is hit, change one of the letters
    in the new individual at random.

At present the population size is set to 10000 and the mutation chance is 1%.
The alphabet is all upper- and lower-case latin chars and space.

I haven’t done any concurrency yet, but shouldn’t be hard. Still, it looks
like it does over 100 generations per second on my Mac in `--release` mode.
