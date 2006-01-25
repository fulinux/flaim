//------------------------------------------------------------------------------
// Desc:	Random number routines
//
// Tabs:	3
//
//		Copyright (c) 1995-1998, 2000, 2003,2005-2006 Novell, Inc. All Rights Reserved.
//
//		This program is free software; you can redistribute it and/or
//		modify it under the terms of version 2 of the GNU General Public
//		License as published by the Free Software Foundation.
//
//		This program is distributed in the hope that it will be useful,
//		but WITHOUT ANY WARRANTY; without even the implied warranty of
//		MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//		GNU General Public License for more details.
//
//		You should have received a copy of the GNU General Public License
//		along with this program; if not, contact Novell, Inc.
//
//		To contact Novell about this file by physical or electronic mail,
//		you may find current contact information at www.novell.com
//
// $Id: ftkrand.cpp 3115 2006-01-19 13:24:39 -0700 (Thu, 19 Jan 2006) dsanders $
//------------------------------------------------------------------------------

#include "flaimsys.h"

/****************************************************************************
	This random number generator is based on Park & Miller's "suggested
minimal standard" for random number generation, pp 1192-1201 of the Oct 1988
issue of _Communications_of_the_ACM_ (vol 31 number 10).  It is a Lehmer
generator, which are of the form x[n] = A * x[n-1] % M, with A and M being
judiciously chosen constants.  More formally, this is a "prime modulus
multiplicative linear congruential generator," or PMMLCG.
	Park & Miller originally suggested A==16807 and M==2**31-1, but an update
in CACM Vol 36, No. 7 pp 108-110 (July 1993) indicates that they have found
a better multiplier (A == 48271) to use with the same modulus (2**31-1).
This implementation uses the updated multiplier.
	To quote Park & Miller 1988, "We believe that this is the generator that
should always be used--unless one has access to a random number generator
KNOWN to be better."
	This algorithm produces a full-period generator; that is, starting from
any seed between 1 and 2**31-2, it generates all other values between 1
and 2**31-2 before it returns to the starting point -- whereupon it repeats
the same sequence of 31-bit values.  This is true for either choice of A
(16807 or 48271).
	The July 1993 article includes criticism by George Marsaglia of the Park
and Miller generator.  Marsaglia feels that longer periods are needed.  For
a description of his "subtract-with-borrow" (SWB) generators, see "A New
Class of Random Number Generators", The Annals of Applied Probability,
(1991) Vol. 1, No. 3, pp. 462-480.  These generators require more state
information (~48 longwords) but produce generators with periods on the
order of 10**445. They also pass more stringent tests than the congruential
generators, and so might be considered 'a random number generator KNOWN to
be better.' However, Marsaglia does not spell out all the details needed to
implement SWB, nor does he give any simple test to determine whether an SWB
implementation is correct.
****************************************************************************/

/*************************************************************************
Desc:	Set the seed from the date and time
*************************************************************************/
void	F_RandomGenerator::randomize( void)
{
	FLMUINT	uiTime;

	f_timeGetSeconds( &uiTime );
	randomSetSeed( (FLMUINT32)(((FLMUINT32)uiTime % MAX_RANDOM) + 1));
}

/*************************************************************************
Desc:	initialize the seed to a known value
*************************************************************************/
void F_RandomGenerator::randomSetSeed(
	FLMINT32				ui32Seed)
{
	register FLMINT32 	i32Seed = (FLMINT32)ui32Seed;

	if( i32Seed > 0 && i32Seed <= MAX_RANDOM)
	{
		m_i32Seed = i32Seed;
	}
	else
	{
		randomSetSeed( (FLMUINT32) 
			(i32Seed < 1
				? i32Seed + MAX_RANDOM
				: i32Seed - MAX_RANDOM));
	}
}

/*************************************************************************
Desc:	Generate the next number in the pseudo-random sequence
		i.e.,	"f_randomLong( &r) > MAX_RANDOM/2" will be true half the
		time,	on average.  Likewise, "f_randomLong( &r) & 0x1" has a 50-50
		chance of being true.
*************************************************************************/
FLMINT32 F_RandomGenerator::randomLong( void)
{

#define M		2147483647		/* PMMLCG modulus == MAX_RANDOM + 1	  	*/
#define A		48271				/* PMMLCG multiplier									*/
#define CHECK	399268537		/* produced by 10000 iterations from seed of 1 */

	register FLMUINT32 hi;
	register FLMUINT32 lo;
	register FLMUINT32 ui32Seed = m_i32Seed; /* input is 31-bit number */

	hi = (ui32Seed >> 16);				/* hi = a (high-order 15 bits of x[n-1]) */
	lo = ui32Seed & 0xFFFF;				/* lo = b (low-order 16 bits of x[n-1]) */
	lo *= A;								/* lo = c * b = d:e (16:16 bits = 32 bits) */
	hi *= A;								/* hi = c * a = f:g (15:16 bits = 31 bits) */

	hi += (lo >> 16) & 0xFFFF;		/* hi = f:g + d = h (31 bits) */
	lo &= 0xFFFF;						/* lo = e (16 bits) */

		/*
		* Now, the 'longhand' product has been calculated.  It is stored in
		* hi:lo (31:16 bits) = h:e (31:16 bits).
		*
		* Now, redistribute the number h:e (31:16 bits) into x:y (16:31 bits)
		*/

	lo |= (hi & 0x7FFF) << 16;		/* lo = y = (low 15 bits of h spliced into e) */
	hi >>= 15;							/* hi = x (high 16 bits of h) */
	lo += hi;							/* lo = z = y + x (32 bits) */

		/*
		* At this point, the value has been reduced modulo M to the 32-bit
		* value z, stored in lo.  Reduce if the high-order bit is set.
		*/

	if( lo & 0x80000000L)			/* subtract 2**31 - 1 if necessary */
	{
		lo &= 0x7FFFFFFF;				/* equivalent to lo = lo - 2**31 */
		lo++;								/* equivalent to lo = lo + 1	*/
	}

		/* we don't need to worry about lo == M, because it can't happen */

	return( m_i32Seed = lo);
}

/*************************************************************************
Desc:	return a random integer between lo and hi, inclusive.
		(where lo and hi are integer arguments).
Example:
		The code "RandomChoice( &r, 1, 6) + RandomChoice( &r, 1, 6)" will
		simulate the roll of a standard 6-sided die.
Note:	The distance (range) between lo and hi must be no greater than
		MAX_RANDOM.  Normally, RandomChoice computes its answer by taking
		a f_randomLong modulo the desired range.  If the range is large enough,
		aliasing effects would cause some answers to be produced too often.
		Therefore, f_randomChoice uses a better but slower algorithm if the
		range is >= 1 Meg (2**20).
*************************************************************************/
FLMINT32 F_RandomGenerator::randomChoice(
	FLMINT32				lo,		/* lowest allowed return value */
	FLMINT32				hi			/* highest allowed return value */
	)
{
	register FLMINT32 range = hi - lo + 1;

	if( range < (1L << 20))
	{
		return( lo + randomLong() % range);
	}
	else
	{
		register FLMINT32 mask = 0;
		register FLMINT32 x;

		range--;
		for( x = range; x > 0; x >>= 1)
		{
			mask = (mask << 1) | 1;
		}

		do
		{
			x = randomLong() & mask;
		}	while( x > range);

		return( lo + x);
	}
}


/*************************************************************************
Desc:	Return TRUE a certain percentage of the time
Example:
		This code will decimate a population (that is, it will kill 10% of
		the "life_force" group):

			for( i=0; i<MAX; i++)
				if( RandomTruth( &r, 10))
					life_force[ i] = 0;

*************************************************************************/
FLMINT F_RandomGenerator::randomTruth(
	FLMINT					iPercentageTrue		/* 1 <= int <= 100 */
	)
{
	return( randomChoice( 1, 100) <= iPercentageTrue);
}

