//-------------------------------------------------------------------------
// Desc:	Unset packing for FLAIM structures.
// Tabs:	3
//
//		Copyright (c) 2002-2006 Novell, Inc. All Rights Reserved.
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
// $Id: fpackoff.h 12267 2006-01-19 14:46:28 -0700 (Thu, 19 Jan 2006) dsanders $
//-------------------------------------------------------------------------

// IMPORTANT NOTE: DO NOT put #ifdef FPACKOFF_H in this file!  We want
// to be able to include it in many different places.

#if !defined(FLM_UNIX) && !defined( FLM_64BIT)
	#pragma pack(pop)
#endif
