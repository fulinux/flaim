//-------------------------------------------------------------------------
// Desc:	Random number generation.
// Tabs:	3
//
//		Copyright (c) 1993-2000,2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: ftkrand.cpp 12299 2006-01-19 15:01:23 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

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
FLMEXP void FLMAPI f_randomize(
	f_randomGenerator *	pRand)
{
	FLMUINT	uiTime;

	f_timeGetSeconds( &uiTime );
	f_randomSetSeed( pRand, (FLMUINT32)(((FLMUINT32)uiTime % MAX_RANDOM) + 1));

	return;
}

/*************************************************************************
Desc:	initialize the seed to a known value
*************************************************************************/
FLMEXP void FLMAPI f_randomSetSeed(
	f_randomGenerator *	pRand,
	FLMINT32					ui32Seed)
{
	register FLMINT32 i32Seed = (FLMINT32) ui32Seed;

		/* fix seed so that it is in legal range */

	if( i32Seed > 0 && i32Seed <= MAX_RANDOM)
	{
		pRand->i32Seed = i32Seed;
	}
	else
	{
		f_randomSetSeed( pRand, (FLMINT32)(i32Seed < 1
											? i32Seed + MAX_RANDOM
											: i32Seed - MAX_RANDOM));
	}
}

/*************************************************************************
Desc:	Generate the next number in the pseudo-random sequence
		i.e.,	"f_randomLong( &r) > MAX_RANDOM/2" will be true half the
		time,	on average.  Likewise, "f_randomLong( &r) & 0x1" has a 50-50
		chance of being true.
Note:
			a  b	(b is the lower 16 bits of x[n-1]; a is the upper 15 bits)
	x			c	(sixteen-bit value of A)
	------------
			d  e	(== c * b; d and e are each 16 bits)
	+	f  g		(== c * a; f is 15 bits and g is 16 bits)
	------------
			h  e	(h:e is 31:16 bits, where h is the 31 bit sum f:g + d)

	It might appear that h could be as wide as 32 bits, but since overall we're
	doing a 16-bit x 31-bit = 47-bit product; h:e must fit in 31:16 bits.
	The mod-by-M operation is then performed by splitting the 47-bit product
	h:e into the upper 16 bits ('x') and the lower 31 bits ('y') and applying
	the slick mod trick described below.

	The slick mod operation is performed by splitting the 47-bit product z
	into the upper 16 bits ('x') and the lower 31 bits ('y').  Then
	  z % M = x:y % M
			  = (x * 2**31 + y) % M
	Rewriting 2**31 == 2**31 - 1 + 1 == M + 1, produces
			  = ((x * (M+1) + y) % M
			  = (x*M + x + y) % M
	But x*M == 0 (mod M), so we can cancel the first term, producing
			  = (x + y) % M.
	Now, if x+y < M, then x+y == (x+y) % M, and the answer has been found.
	f x+y > M, then the 16-bit + 31-bit sum overflowed into the upper bit.
	But we can apply the trick again: z % M = x:y % M = (x+y) % M.  In this
	case, however, x is the single high-order bit, and y is the low-order 31
	bits.  Thus x == 1 and y == z & 0x7FFFFFFF, so x+y can be computed by
	clearing the high order bit of z and incrementing the result.  Viewed
	another way, this operation simply subtracts M: clearing the high-order
	bit is equivalent to subtracting 2**31; incrementing the result means
	that 2**31-1 has been subtracted instead.
		Though it might appear that the result might overflow into the high-order
	bit, this can't really happen.  The 32-bit value was computed by adding a 16
	bit value to a 31-bit value, so the largest possible result is 0xFFFF
	+ 0x7FFFFFFF = 0x8000FFFE = 2**31 + 0xFFFE.  After subtracting 2**31-1,
	the result cannot be any larger than 0xFFFF.
		Once the high-order bit is clear the answer is less than or equal to M.
	If it is less than M, we're done; if it is equal to M, then the correct
	result is zero (M % M == 0).
		However, for this application it is unnecessary to handle the case where
	the answer is zero, because the 47-bit product is never a multiple of M
	(if the input seed is valid -- that is, in the range 1..0x7FFFFFFE).
	Proof (by contradiction): Let x be the previous seed, and let z be the
	47-bit product. Then z = A * x.  If M divides the product z, then M must
	divide either A or x (or both), because M is prime.  But M cannot divide
	either: both values are greater than one and less than M.  -><-
		As corroborating evidence, we have the fact that the generator, if
	correctly implemented, never generates the value 0; its results are always
	in the range 1..0x7FFFFFFE (according to Park & Miller).  If it ever were
	to	produce the value zero, it would be a particularly BAD random number
	generator since it would continue to generate nothing but zero from
	that point on (it IS a multiplicative generator, after all).
*************************************************************************/
FLMEXP FLMINT32 FLMAPI f_randomLong(
	f_randomGenerator *	generator)
{
#define M		2147483647
#define A		48271
#define CHECK	399268537

	register FLMUINT32 hi;
	register FLMUINT32 lo;
	register FLMUINT32 ui32Seed =generator->i32Seed; /* input is 31-bit number */

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

	if( lo & 0x80000000)				/* subtract 2**31 - 1 if necessary */
	{
		lo &= 0x7FFFFFFF;				/* equivalent to lo = lo - 2**31 */
		lo++;								/* equivalent to lo = lo + 1	*/
	}

	/* we don't need to worry about lo == M, because it can't happen */

	return( generator->i32Seed = lo);
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
FLMEXP FLMINT32 FLMAPI f_randomChoice(
	f_randomGenerator *	r,
	FLMINT32				lo,		/* lowest allowed return value */
	FLMINT32				hi			/* highest allowed return value */
	)
{
	register FLMINT32 range = hi - lo + 1;

	if( range < (1L << 20))
	{
		return( lo + f_randomLong( r) % range);
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
			x = f_randomLong( r) & mask;
		}	while( x > range);

		return( lo + x);
	}
}


/*************************************************************************
Desc:	return TRUE a certain percentage of the time
Example:
		This code will decimate a population (that is, it will kill 10% of
		the "life_force" group):

			for( i=0; i<MAX; i++)
				if( RandomTruth( &r, 10))
					life_force[ i] = 0;

*************************************************************************/
FLMEXP FLMINT FLMAPI f_randomTruth(
	f_randomGenerator  *	pRand,
	FLMINT					iPercentageTrue		/* 1 <= int <= 100 */
	)
{
	return( f_randomChoice( pRand, 1, 100) <= iPercentageTrue);
}
