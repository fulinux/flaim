//-------------------------------------------------------------------------
// Desc:	Cross platform toolkit for time functions.
// Tabs:	3
//
//		Copyright (c) 2000,2002-2003,2005-2006 Novell, Inc. All Rights Reserved.
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
// $Id: ftktime.cpp 12299 2006-01-19 15:01:23 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

#include "flaimsys.h"

#ifdef FLM_NLM

	#if !defined( __MWERKS__)
		#pragma warning 007 9
	#endif

	FINLINE unsigned long time(
		unsigned long *	pulSeconds)
	{
		clockAndStatus clock;

		GetCurrentClock( &clock);

		if (pulSeconds)
		{
			*pulSeconds = (unsigned long)clock [0];
		}

		return ( (unsigned long)clock[0] );
	}
#endif

#define	BASEYR			1970				/* all gmt calcs done since 1970		*/
#define	SECONDSPERDAY	86400l			/* 24 hours * 60 minutes * 60 seconds */
#define	SECONDSPERHOUR	3600				/* 60 minutes * 60 seconds 			*/
#define	DDAYSPERYEAR	365				/* 365 days/year							*/

static FLMUINT8 ui8NumDaysPerMonth[2][12] = {    /* days of the months */
	{31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
	{31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31} };

static FLMUINT16 ui16NumDaysFromJan1st[2][12] = {
	/* current total of the days in the year by mon */
	{ 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334},
	{ 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335} };

static F_TMSTAMP maxdate =
	{ 2106, 1, 6, 11, 0, 0, 0 };

static FLMUINT f_timeLeapYearsSince1970(
	FLMUINT16 year);

#define f_timeIsLeapYear( year) \
	((((year) & 0x03) == 0) && (((year) % 100) != 0) || \
		(((year) % 400) == 0))

/****************************************************************************
Desc:		Gets the number of seconds since 1980 or 1970.
****************************************************************************/
FLMEXP void FLMAPI f_timeGetSeconds(
	FLMUINT	*		puiSeconds)
{
#if defined( FLM_WIN)
	*puiSeconds = (FLMUINT) time( (time_t *) 0 );

#elif defined( FLM_NLM)
	*puiSeconds = (FLMUINT) time( NULL);

#elif defined( FLM_UNIX)
	*puiSeconds = (FLMUINT) time( 0);
	
#else
#error Platform not supported
#endif
}

/****************************************************************************
Desc:		Gets the time stamp from the system clock.
****************************************************************************/
FLMEXP void FLMAPI f_timeGetTimeStamp(
	F_TMSTAMP *		pTimeStamp)
{
#if defined( FLM_WIN)
	SYSTEMTIME	rightnow;
	
	GetLocalTime( &rightnow );

	pTimeStamp->year  = rightnow.wYear;
	pTimeStamp->month = (FLMUINT8)(rightnow.wMonth - 1);
	pTimeStamp->day   = (FLMUINT8)rightnow.wDay;

	pTimeStamp->hour   = (FLMUINT8)rightnow.wHour;
	pTimeStamp->minute = (FLMUINT8)rightnow.wMinute;
	pTimeStamp->second = (FLMUINT8)rightnow.wSecond;
	pTimeStamp->hundredth = rightnow.wMilliseconds / 10;

#elif defined( FLM_NLM)
	f_timeSecondsToDate( (FLMUINT)
		time( NULL) - f_timeGetLocalOffset(), pTimeStamp);

#elif defined( FLM_UNIX)
	time_t now;
	struct tm rightnow;

	now = time( (time_t *) 0 );
	(void)localtime_r( &now, &rightnow );

	pTimeStamp->year        = rightnow.tm_year + 1900;
	pTimeStamp->month       = rightnow.tm_mon;
	pTimeStamp->day         = rightnow.tm_mday;
	pTimeStamp->hour        = rightnow.tm_hour;
	pTimeStamp->minute      = rightnow.tm_min;
	pTimeStamp->second      = rightnow.tm_sec;
	pTimeStamp->hundredth   = 0;
#else
#error Platform not supported
#endif
}

/****************************************************************************
Desc:		Returns the local time bias in seconds
****************************************************************************/
FLMEXP FLMINT FLMAPI f_timeGetLocalOffset( void)
{
	FLMINT		iOffset = 0;

#if defined( FLM_WIN)
	TIME_ZONE_INFORMATION		tzInfo;
	DWORD								retVal;

	retVal = GetTimeZoneInformation( &tzInfo);

	if( retVal != TIME_ZONE_ID_UNKNOWN)
	{
		iOffset = 
			(retVal == TIME_ZONE_ID_DAYLIGHT && tzInfo.DaylightDate.wMonth
				? tzInfo.Bias + tzInfo.DaylightBias
				: tzInfo.Bias) * 60;
	}

#elif defined( FLM_NLM)

	Synchronized_Clock_T    SynchronizedClock;

	f_memset( &SynchronizedClock, 0, sizeof( SynchronizedClock));
	GetSyncClockFields(
		SYNCCLOCK_DAYLIGHT_BIT | SYNCCLOCK_DAYLIGHT_OFFSET_BIT |
		SYNCCLOCK_DAYLIGHT_ON_OFF_BIT | SYNCCLOCK_TIMEZONE_OFFSET_BIT,
		&SynchronizedClock);

	iOffset = (FLMINT)SynchronizedClock.timezoneOffset;
	if( SynchronizedClock.daylight && SynchronizedClock.daylightOnOff)
	{
		iOffset += (FLMINT)SynchronizedClock.daylightOffset;
	}
#elif defined( FLM_UNIX)
	time_t		gmtTime;
	time_t		localTime;
	struct tm	gmtTm;
	
	gmtTime = time( (time_t *)0);
	gmtime_r( &gmtTime, &gmtTm);
	localTime = mktime( &gmtTm);
	iOffset = (FLMINT)((FLMINT64)localTime - (FLMINT64)gmtTime);

#else
#error Platform not supported
#endif

	return( iOffset);
}

/****************************************************************************
Desc:		Count the number of leap years from 1970 to given year.
Notes:	According to the Gregorian calendar (which we currently use), the
			year is a leap year if it is divisible by 4, unless it is a century
			year, then it must be divisible by 400.
****************************************************************************/
static FLMUINT f_timeLeapYearsSince1970(
	FLMUINT16	ui16Year)
{
	FLMUINT		uiTemp;

	/* first calculate # of leap years since 1600 */

	ui16Year -= 1601;								/* ui16Year = number of years since 1600*/
	uiTemp = (										/* Count leap years						*/
		(ui16Year / 4) -							/* Count potential leap years			*/
		(ui16Year / 100) +							/* Subtract out century years			*/
		(ui16Year / 400) +							/* Add back in quadricentenial years*/
		1											/* And don't forget to count 1600	*/
	);

	/* now subtract # of leap years between 1600 and 1970 */
	/* (the following becomes a constant at compile time) */

	uiTemp -= ((BASEYR-1600) / 4) - ((BASEYR-1600) / 100) + 1;
	return(uiTemp);
}

/****************************************************************************
Desc:		Convert from seconds to the F_TMSTAMP structure.
****************************************************************************/
FLMEXP void FLMAPI f_timeSecondsToDate(
	FLMUINT 			uiSeconds,
	F_TMSTAMP *		date)
{
	FLMUINT			uiLeapYear;
	FLMUINT			uiMonth;
	FLMUINT			uiDaysInMonth;
	FLMUINT			uiDay;

	uiDay = uiSeconds / SECONDSPERDAY;					// # of days since 1970
	date->year = (FLMUINT16)((uiDay / DDAYSPERYEAR)	+ BASEYR);
	uiDay = uiDay % DDAYSPERYEAR;							// # of days into year

	/*
	Check to see that the value for the current day is greater than the 
	number of leap years since 1970.  This is because we will be 
	subtracting the leap days from the current day and we don't want 
	the value for the day to go negative.
	*/

	while( uiDay < f_timeLeapYearsSince1970(date->year)) // if day < # of leap years
	{
		date->year--;											// decrement the year
		uiDay += DDAYSPERYEAR;								// adjust day by days/year
	}

	uiDay -= f_timeLeapYearsSince1970( date->year);	// subtract leap days
	uiLeapYear = f_timeIsLeapYear( date->year );		// set leap year flag

	/*
	Find what our offset into the current month is. 
	To do this, we subtract out the number of days for each month, until
	the number of days left does not span the end of the current month
	*/

	for( uiMonth = 0; 
		  uiMonth < 12 && 
			(uiDay >= (uiDaysInMonth = ui8NumDaysPerMonth[uiLeapYear][uiMonth])); 
		  uiMonth++)
	{
		uiDay -= uiDaysInMonth;								// subtract days in month
	}
	date->month = (FLMUINT8) uiMonth;					// set month, day
	date->day = (FLMUINT8) (++uiDay);

	uiDay = uiSeconds % SECONDSPERDAY;					// mod by seconds/day
	date->hour = (FLMUINT8)(uiDay / SECONDSPERHOUR);// get # of hours
	uiDay = uiDay % SECONDSPERHOUR;		
	date->minute = (FLMUINT8)(uiDay / 60);				// get # of minutes
	date->second = (FLMUINT8)(uiDay % 60);
	date->hundredth = 0;										// no fractional seconds
}

/****************************************************************************
Desc:		Convert a time stamp to the number of seconds.
****************************************************************************/
FLMEXP void FLMAPI f_timeDateToSeconds(
	F_TMSTAMP *		pTimeStamp,			// [in] - time stamp of date
	FLMUINT *		puiSeconds)			// [out] - seconds of time stamp
{

	FLMUINT			uiDays = 0;

	// is date past max?
	if( f_timeCompareTimeStamps( pTimeStamp, &maxdate, COMPARE_DATE_AND_TIME) > 0)
	{
			*pTimeStamp = maxdate;
	}

	// Do date portion of calculation - result is days since 1/1/1970.

	if( pTimeStamp->year) 
	{
		uiDays = 
			(pTimeStamp->year - BASEYR) * 365 +	// years since BASE * days
			f_timeLeapYearsSince1970( pTimeStamp->year) +// leap years since BASE
			ui16NumDaysFromJan1st[ f_timeIsLeapYear(pTimeStamp->year)][pTimeStamp->month] +
			pTimeStamp->day - 1;						// days since 1st of month
	}

	//	Do time part of calculation - secs since 1/1/1970 12:00am.

	*puiSeconds = (((uiDays * 24) +				// convert days to hours
		pTimeStamp->hour ) * 60	+					// convert hours to min
		pTimeStamp->minute) * 60	+				// convert min to sec
		pTimeStamp->second;							// give secs granularity

}

/****************************************************************************
Desc:	Compare two time stamps
****************************************************************************/
FLMEXP FLMINT FLMAPI f_timeCompareTimeStamps(
	F_TMSTAMP *		pTimeStamp1,
	F_TMSTAMP *		pTimeStamp2,
	FLMUINT			uiCompareFlag)
{
	if( uiCompareFlag != COMPARE_TIME_ONLY)				/* not comparing times only	*/
	{
		if( pTimeStamp1->year != pTimeStamp2->year) 
		{
			return((pTimeStamp1->year < pTimeStamp2->year) ? -1 : 1);
		}
		if( pTimeStamp1->month != pTimeStamp2->month) 
		{
			return((pTimeStamp1->month < pTimeStamp2->month) ? -1 : 1);
		}
		if( pTimeStamp1->day != pTimeStamp2->day) 
		{
			return((pTimeStamp1->day < pTimeStamp2->day) ? -1 : 1);
		}
	}
	if( uiCompareFlag != COMPARE_DATE_ONLY)
	{
		if( pTimeStamp1->hour != pTimeStamp2->hour) 	
		{
			return((pTimeStamp1->hour < pTimeStamp2->hour) ? -1 : 1);
		}
		if( pTimeStamp1->minute != pTimeStamp2->minute) 		
		{
			return((pTimeStamp1->minute < pTimeStamp2->minute) ? -1 : 1);	
		}
		if( pTimeStamp1->second != pTimeStamp2->second) 		
		{
			return((pTimeStamp1->second < pTimeStamp2->second) ? -1 : 1);	
		}
	}
	return(0);
}

/****************************************************************************
Desc:		Get the current time in milliseconds.
****************************************************************************/
#if defined( FLM_UNIX)
FLMEXP FLMUINT FLMAPI f_timeGetMilliTime()
{
#ifdef FLM_SOLARIS
	static hrtime_t epoch = 0;
	hrtime_t now = gethrtime();  /* hrtime returns in nanosecs */

	if (!epoch)
		epoch = now;

	return( (FLMUINT)((now - epoch) / (1000 * 1000)));
#else
	static int epoch = 0;
	struct timeval tv;

	gettimeofday(&tv, 0);
	if (!epoch)
		epoch = tv.tv_sec;
	
	return( (FLMUINT)((tv.tv_sec - epoch) * 1000 + tv.tv_usec / 1000));
#endif
}
#endif

/****************************************************************************
Desc:	Get the current time as platform-dependent timer units.
****************************************************************************/
FLMEXP FLMUINT FLMAPI f_getCurrTimeAsTimerUnits( void)
{
	return( FLM_GET_TIMER());
}

/****************************************************************************
Desc:	Convert seconds to platform-dependent timer units.
****************************************************************************/
FLMEXP FLMUINT FLMAPI f_secondsToTimerUnits(
	FLMUINT		uiSeconds)
{
	FLMUINT	uiTimerUnits;

	FLM_SECS_TO_TIMER_UNITS( uiSeconds, uiTimerUnits);
	return( uiTimerUnits);
}

/****************************************************************************
Desc:	Convert platform-dependent timer units to seconds.
****************************************************************************/
FLMEXP FLMUINT FLMAPI f_timerUnitsToSeconds(
	FLMUINT		uiTimerUnits)
{
	FLMUINT	uiSeconds;

	FLM_TIMER_UNITS_TO_SECS( uiTimerUnits, uiSeconds);
	return( uiSeconds);
}

/****************************************************************************
Desc:	Convert milliseconds to platform-dependent timer units.
****************************************************************************/
FLMEXP FLMUINT FLMAPI f_milliSecondsToTimerUnits(
	FLMUINT		uiMilliSeconds)
{
	FLMUINT	uiTimerUnits;

	FLM_MILLI_TO_TIMER_UNITS( uiMilliSeconds, uiTimerUnits);
	return( uiTimerUnits);
}

/****************************************************************************
Desc:	Convert platform-dependent timer units to milli-seconds.
****************************************************************************/
FLMEXP FLMUINT FLMAPI f_timerUnitsToMilliSeconds(
	FLMUINT		uiTimerUnits)
{
	FLMUINT	uiMilliSeconds;

	FLM_TIMER_UNITS_TO_MILLI( uiTimerUnits, uiMilliSeconds);
	return( uiMilliSeconds);
}

/****************************************************************************
Desc:	Return elapsed time (as platform-dependent timer units).  Input
		parameters must be passed in as platform-dependent timer units.
****************************************************************************/
FLMEXP FLMUINT FLMAPI f_elapsedTimeTimerUnits(
	FLMUINT	uiEarlierTimeTimerUnits,
	FLMUINT	uiLaterTimeTimerUnits)
{
	return( FLM_ELAPSED_TIME( uiLaterTimeTimerUnits, uiEarlierTimeTimerUnits));
}
